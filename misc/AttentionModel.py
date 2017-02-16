from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import vgg
import copy

import numpy as np
import misc.utils as utils

# The maximimum step during generation
MAX_STEPS = 30

class AttentionModel():
    """
    This model is not using the show attend tell algorithm, but given seq2seq attention decoder.
    """

    def initialize(self, sess):
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        # Initialize the saver
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        # Load weights from the checkpoint
        if vars(self.opt).get('start_from', None):
            self.saver.restore(sess, self.opt.ckpt.model_checkpoint_path)
        # Initialize the summary writer
        self.summary_writer = tf.summary.FileWriter(self.opt.checkpoint_path, sess.graph)

    def __init__(self, opt):
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.vocab_size = opt.vocab_size
        self.seq_per_img = opt.seq_per_img

        self.opt = opt

        # Variable indicating in training mode or evaluation mode
        self.training = tf.Variable(True, trainable = False, name = "training")

        # Input variables
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], name = "images")
        self.labels = tf.placeholder(tf.int32, [None, self.seq_length + 2], name = "labels")
        self.masks = tf.placeholder(tf.float32, [None, self.seq_length + 2], name = "masks")

        # Build CNN
        if vars(self.opt).get('start_from', None):
            cnn_weight = None
        else:
            cnn_weight = self.opt.cnn_weight
        if self.opt.cnn_model == 'vgg16':
            self.cnn = vgg.Vgg16(cnn_weight)
        if self.opt.cnn_model == 'vgg19':
            self.cnn = vgg.Vgg19(cnn_weight)
            
        with tf.variable_scope("cnn"):
            self.cnn.build(self.images)

        if self.opt.cnn_model == 'vgg16':
            self.context = self.cnn.conv5_3
        if self.opt.cnn_model == 'vgg19':
            self.context = self.cnn.conv5_4
        
        self.cnn_training = self.cnn.training

        # Variable in language model
        with tf.variable_scope("rnnlm"):
            # Word Embedding table
            self.Wemb = tf.Variable(tf.random_uniform([self.vocab_size + 1, self.input_encoding_size], -0.1, 0.1), name='Wemb')

            # RNN cell
            if opt.rnn_type == 'rnn':
                self.cell_fn = cell_fn = tf.contrib.rnn.BasicRNNCell
            elif opt.rnn_type == 'gru':
                self.cell_fn = cell_fn = tf.contrib.rnn.GRUCell
            elif opt.rnn_type == 'lstm':
                self.cell_fn = cell_fn = tf.contrib.rnn.LSTMCell
            else:
                raise Exception("RNN type not supported: {}".format(opt.rnn_type))
            
            # keep_prob is a function of training flag
            self.keep_prob = tf.cond(self.training, 
                                lambda : tf.constant(1 - self.drop_prob_lm),
                                lambda : tf.constant(1.0), name = 'keep_prob')
            # basic cell has dropout wrapper
            self.basic_cell = cell = tf.contrib.rnn.DropoutWrapper(cell_fn(self.rnn_size, state_is_tuple = True), 1.0, self.keep_prob)
            # cell is the final cell of each timestep
            self.cell = tf.contrib.rnn.MultiRNNCell([cell] * opt.num_layers, state_is_tuple = True)

    def build_model(self):
        with tf.name_scope("batch_size"):
            # Get batch_size from the first dimension of self.images
            self.batch_size = tf.shape(self.images)[0]
        with tf.variable_scope("rnnlm"):
            flattened_ctx = tf.reshape(self.context, [self.batch_size, 196, 512])
            ctx_mean = tf.reduce_mean(flattened_ctx, 1)
            
            # Initialize the first hidden state with the mean context
            initial_state = utils.get_initial_state(ctx_mean, self.cell.state_size)
            # Replicate self.seq_per_img times for each state and image embedding
            self.initial_state = initial_state = utils.expand_feat(initial_state, self.seq_per_img)
            self.flattened_ctx = flattened_ctx = tf.reshape(tf.tile(tf.expand_dims(flattened_ctx, 1), [1, self.seq_per_img, 1, 1]), 
                [self.batch_size * self.seq_per_img, 196, 512])

            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.seq_length + 1, value=tf.nn.embedding_lookup(self.Wemb, self.labels[:,:self.seq_length + 1]))
            rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]

            outputs, last_state = tf.contrib.legacy_seq2seq.attention_decoder(rnn_inputs, initial_state, flattened_ctx, self.cell, loop_function=None)
            outputs = tf.concat(axis=0, values=outputs)

            self.logits = slim.fully_connected(outputs, self.vocab_size + 1, activation_fn = None, scope = 'logit')
            self.logits = tf.split(axis=0, num_or_size_splits=len(rnn_inputs), value=self.logits)

        with tf.variable_scope("loss"):
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(self.logits,
                    [tf.squeeze(label, [1]) for label in tf.split(axis=1, num_or_size_splits=self.seq_length + 1, value=self.labels[:, 1:])], # self.labels[:,1:] is the target
                    [tf.squeeze(mask, [1]) for mask in tf.split(axis=1, num_or_size_splits=self.seq_length + 1, value=self.masks[:, 1:])])
            self.cost = tf.reduce_mean(loss)

        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.cnn_lr = tf.Variable(0.0, trainable=False)

        # Collect the rnn variables, and create the optimizer of rnn
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnnlm')
        grads = utils.clip_by_value(tf.gradients(self.cost, tvars), -self.opt.grad_clip, self.opt.grad_clip)
        #grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
        #        self.opt.grad_clip)
        optimizer = utils.get_optimizer(self.opt, self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Collect the cnn variables, and create the optimizer of cnn
        cnn_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn')
        cnn_grads = utils.clip_by_value(tf.gradients(self.cost, cnn_tvars), -self.opt.grad_clip, self.opt.grad_clip)
        #cnn_grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, cnn_tvars),
        #        self.opt.grad_clip)
        cnn_optimizer = utils.get_cnn_optimizer(self.opt, self.cnn_lr)
        self.cnn_train_op = cnn_optimizer.apply_gradients(zip(cnn_grads, cnn_tvars))

        tf.summary.scalar('training loss', self.cost)
        tf.summary.scalar('learning rate', self.lr)
        tf.summary.scalar('cnn learning rate', self.cnn_lr)
        self.summaries = tf.summary.merge_all()

    def build_generator(self):
        """
        Generator for generating captions
        Support sample max or sample from distribution
        No Beam search here; beam search is in decoder
        """
        # Variables for the sample setting
        self.sample_max = tf.Variable(True, trainable = False, name = "sample_max")
        self.sample_temperature = tf.Variable(1.0, trainable = False, name = "temperature")

        self.generator = []
        with tf.variable_scope("rnnlm") as rnnlm_scope:
            flattened_ctx = tf.reshape(self.context, [self.batch_size, 196, 512])
            ctx_mean = tf.reduce_mean(flattened_ctx, 1)

            tf.get_variable_scope().reuse_variables()

            initial_state = utils.get_initial_state(ctx_mean, self.cell.state_size)

            rnn_inputs = [tf.nn.embedding_lookup(self.Wemb, tf.zeros([self.batch_size], tf.int32))] + [0] * (MAX_STEPS - 1)

            # Always pick the word with largest probability as the input of next time step
            def loop(prev, i):
                with tf.variable_scope(rnnlm_scope):
                    prev = slim.fully_connected(prev, self.vocab_size + 1, activation_fn = None, scope = 'logit')                
                    prev_symbol = tf.stop_gradient(tf.cond(self.sample_max,
                        lambda: tf.argmax(prev, 1), # pick the word with largest probability as the input of next time step
                        lambda: tf.squeeze(
                            tf.multinomial(tf.nn.log_softmax(prev) / self.sample_temperature, 1), 1))) # Sample from the distribution
                    self.generator.append(prev_symbol)
                    return tf.nn.embedding_lookup(self.Wemb, prev_symbol)

            outputs, last_state = tf.contrib.legacy_seq2seq.attention_decoder(rnn_inputs, initial_state, flattened_ctx, self.cell, loop_function=loop)
            self.g_outputs = outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size]) 
            self.g_logits = logits = slim.fully_connected(outputs, self.vocab_size + 1, activation_fn = None, scope = 'logit')
            self.g_probs = probs = tf.reshape(tf.nn.softmax(logits), [self.batch_size, MAX_STEPS, self.vocab_size + 1])

        self.generator = tf.transpose(tf.reshape(tf.concat(axis=0, values=self.generator), [MAX_STEPS - 1, -1]))
    
    def build_decoder_rnn(self, first_step):
        """
        This function build a decoder
        if first_step is true, the state is initialized by mean context
        if first_step is not true, the states are placeholder, and should be assigned.
        """
        with tf.variable_scope("rnnlm"):
            flattened_ctx = tf.reshape(self.context, [self.batch_size, 196, 512])
            ctx_mean = tf.reduce_mean(flattened_ctx, 1)

            self.decoder_prev_word = tf.placeholder(tf.int32, [None])            
            if first_step:
                rnn_input = tf.nn.embedding_lookup(self.Wemb, tf.zeros([self.batch_size], tf.int32))
            else:
                rnn_input = tf.nn.embedding_lookup(self.Wemb, self.decoder_prev_word)

            tf.get_variable_scope().reuse_variables()
            if not first_step:
                initial_state = utils.get_placeholder_state(self.cell.state_size)
                self.decoder_flattened_state = utils.flatten_state(initial_state)
            else:
                initial_state = utils.get_initial_state(ctx_mean, self.cell.state_size)

            outputs, state = tf.contrib.legacy_seq2seq.attention_decoder([rnn_input], initial_state, flattened_ctx, self.cell, initial_state_attention = not first_step)
            logits = slim.fully_connected(outputs[0], self.vocab_size + 1, activation_fn = None, scope = 'logit')
            decoder_probs = tf.reshape(tf.nn.softmax(logits), [self.batch_size, self.vocab_size + 1])
            decoder_state = utils.flatten_state(state)

        # output the probability and flattened state to next time step
        return [decoder_probs, decoder_state]


    def build_decoder(self):
        self.decoder_model_init = self.build_decoder_rnn(True) # Used for the first step
        self.decoder_model_cont = self.build_decoder_rnn(False)

    def decode(self, img, beam_size, sess, max_steps=30):
        """Decode an image with a sentences."""
        
        # Initilize beam search variables
        # Candidate will be represented with a dictionary
        #   "indexes": a list with indexes denoted a sentence; 
        #   "words": word in the decoded sentence without <bos>
        #   "score": log-likelihood of the sentence
        #   "state": RNN state when generating the last word of the candidate
        good_sentences = [] # store sentences already ended with <eos>
        cur_best_cand = [] # store current best candidates
        highest_score = 0.0 # hightest log-likelihodd in good sentences
        
        # Get the initial logit and state
        cand = {'indexes': [], 'score': 0}
        cur_best_cand.append(cand)
            
        # Expand the current best candidates until max_steps or no candidate
        for i in xrange(max_steps + 1):
            # expand candidates
            cand_pool = []
            if i == 0:
                all_probs, all_states = self.get_probs_init(img, sess)
            else:
                states = [np.vstack([cand['state'][i] for cand in cur_best_cand]) for i in xrange(len(cur_best_cand[0]['state']))]
                indexes = [cand['indexes'][-1] for cand in cur_best_cand]
                imgs = np.vstack([img] * len(cur_best_cand))
                all_probs, all_states = self.get_probs_cont(states, imgs, indexes, sess)

            # Construct new beams
            for ind_cand in range(len(cur_best_cand)):
                cand = cur_best_cand[ind_cand]
                probs = all_probs[ind_cand]
                state = [x[ind_cand] for x in all_states]
                
                probs = np.squeeze(probs)
                probs_order = np.argsort(-probs)
                # append new end terminal at the end of this beam
                for ind_b in xrange(beam_size):
                    cand_e = copy.deepcopy(cand)
                    cand_e['indexes'].append(probs_order[ind_b])
                    cand_e['score'] -= np.log(probs[probs_order[ind_b]])
                    cand_e['state'] = state
                    cand_pool.append(cand_e)
            # get best beams
            cur_best_cand = sorted(cand_pool, key=lambda cand: cand['score'])
            cur_best_cand = utils.truncate_list(cur_best_cand, beam_size)

            # move candidates end with <eos> to good_sentences or remove it
            cand_left = []
            for cand in cur_best_cand:
                if len(good_sentences) > beam_size and cand['score'] > highest_score:
                    continue # No need to expand that candidate
                if cand['indexes'][-1] == 0: #end of sentence
                    good_sentences.append(cand)
                    highest_score = max(highest_score, cand['score'])
                else:
                    cand_left.append(cand)
            cur_best_cand = cand_left
            if not cur_best_cand:
                break

        # Add candidate left in cur_best_cand to good sentences 
        for cand in cur_best_cand:
            if len(good_sentences) > beam_size and cand['score'] > highest_score:
                continue
            if cand['indexes'][-1] != 0:
                cand['indexes'].append(0)
            good_sentences.append(cand)
            highest_score = max(highest_score, cand['score'])
            
        # Sort good sentences and return the final list
        good_sentences = sorted(good_sentences, key=lambda cand: cand['score'])
        good_sentences = utils.truncate_list(good_sentences, beam_size)

        return [sent['indexes'] for sent in good_sentences]


    def get_probs_init(self, img, sess):
        """Use the model to get initial logit"""
        m = self.decoder_model_init
        
        probs, state = sess.run(m, {self.images: img})
                                                            
        return (probs, state)
        
    def get_probs_cont(self, prev_state, img, prev_word, sess):
        """Use the model to get continued logit"""
        m = self.decoder_model_cont
        prev_word = np.array(prev_word, dtype='int32')

        placeholders = [self.images, self.decoder_prev_word] + self.decoder_flattened_state
        feeded = [img, prev_word] + prev_state
        
        probs, state = sess.run(m, {placeholders[i]: feeded[i] for i in xrange(len(placeholders))})
                                                            
        return (probs, state)