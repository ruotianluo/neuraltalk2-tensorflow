import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import vgg
import copy

import numpy as np
import misc.utils as utils

class ShowAttendTellModel():

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def initialize(self, sess):
        sess.run(tf.initialize_all_variables())
        # Saver
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        if self.opt.start_from is not None:
            self.saver.restore(sess, self.opt.ckpt.model_checkpoint_path)

        self.summary_writer = tf.train.SummaryWriter(self.opt.checkpoint_path, sess.graph)

    def __init__(self, opt):
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.vocab_size = opt.vocab_size
        self.seq_per_img = opt.seq_per_img
        #self.batch_size = opt.batch_size

        self.opt = opt

        # Variable indicating in training mode or evaluation mode
        self.training = tf.Variable(True, trainable = False, name = "training")

        # Input varaibles
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], name = "images")
        self.labels = tf.placeholder(tf.int32, [None, self.seq_length + 2], name = "labels")
        self.masks = tf.placeholder(tf.float32, [None, self.seq_length + 2], name = "masks")

        # VGG 16
        if self.opt.start_from is not None:
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
            #with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([self.vocab_size + 1, self.input_encoding_size], -0.1, 0.1), name='Wemb')

            #
            self.embed_word_W = tf.Variable(tf.random_uniform([self.rnn_size, self.vocab_size + 1], -0.1, 0.1), name='embed_word_W')
            self.embed_word_b = self.init_bias(self.vocab_size + 1, name='embed_word_b')

            # RNN cell
            if opt.rnn_type == 'rnn':
                self.cell_fn = cell_fn = tf.nn.rnn_cell.BasicRNNCell
            elif opt.rnn_type == 'gru':
                self.cell_fn = cell_fn = tf.nn.rnn_cell.GRUCell
            elif opt.rnn_type == 'lstm':
                self.cell_fn = cell_fn = tf.nn.rnn_cell.LSTMCell
            else:
                raise Exception("RNN type not supported: {}".format(opt.rnn_type))

            self.keep_prob = tf.cond(self.training, 
                                lambda : tf.constant(1 - self.drop_prob_lm),
                                lambda : tf.constant(1.0), name = 'keep_prob')

            self.basic_cell = cell = tf.nn.rnn_cell.DropoutWrapper(cell_fn(self.rnn_size, state_is_tuple = True), 1.0, self.keep_prob)

            self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * opt.num_layers, state_is_tuple = True)

    def get_initial_state(self, input, state_size, scope = 'init_state'):
        with tf.variable_scope(scope):
            if isinstance(state_size, tf.nn.rnn_cell.LSTMStateTuple):
                c = slim.fully_connected(input, state_size.c, activation_fn=tf.nn.tanh, scope='LSTM_c')
                h = slim.fully_connected(input, state_size.h, activation_fn=tf.nn.tanh, scope='LSTM_h')
                return tf.nn.rnn_cell.LSTMStateTuple(c,h)
            elif isinstance(state_size, tuple):
                result = [self.get_initial_state(input, state_size[i], "layer_"+str(i)) for i in xrange(len(state_size))]
                return tuple(result)
            elif isinstance(state_size, int):
                return slim.fully_connected(input, state_size, activation_fn=tf.nn.tanh, scope='state')

    def expand_feat(self, input, scope = 'expand_feat'):
        with tf.variable_scope(scope):
            if isinstance(input, tf.nn.rnn_cell.LSTMStateTuple):
                c = self.expand_feat(input.c, scope='expand_LSTM_c')
                h = self.expand_feat(input.h, scope='expand_LSTM_c')
                return tf.nn.rnn_cell.LSTMStateTuple(c,h)
            elif isinstance(input, tuple):
                result = [self.expand_feat(input[i], "expand_layer_"+str(i)) for i in xrange(len(input))]
                return tuple(result)
            else:
                return tf.reshape(tf.tile(tf.expand_dims(input, 1), [1, self.seq_per_img, 1]), [tf.shape(input)[0] * self.seq_per_img, input.get_shape()[1].value])

    def last_hidden_vec(self, state):
        if isinstance(state, tuple):
            return self.last_hidden_vec(state[len(state) - 1])
        elif isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
            return state.h
        else:
            return state

    def build_model(self):
        with tf.name_scope("batch_size"):
            self.batch_size = tf.shape(self.images)[0]
        with tf.variable_scope("rnnlm"):
            #flattened_ctx = tf.reshape(self.context, [self.batch_size, tf.shape(self.context)[1]*tf.shape(self.context)[2],tf.shape(self.context)[3]])
            flattened_ctx = tf.reshape(self.context, [self.batch_size, 196, 512])
            ctx_mean = tf.reduce_mean(flattened_ctx, 1)

            initial_state = self.get_initial_state(ctx_mean, self.cell.state_size)
            # Replicate self.seq_per_img times for each state and image embedding
            self.initial_state = initial_state = self.expand_feat(initial_state)
            #self.flattened_ctx = flattened_ctx = tf.reshape(tf.tile(tf.expand_dims(flattened_ctx, 1), [1, self.seq_per_img, 1, 1]), 
            #    [self.batch_size * self.seq_per_img, tf.shape(flattened_ctx)[1], tf.shape(flattened_ctx)[2]])
            self.flattened_ctx = flattened_ctx = tf.reshape(tf.tile(tf.expand_dims(flattened_ctx, 1), [1, self.seq_per_img, 1, 1]), 
                [self.batch_size * self.seq_per_img, 196, 512])

            #projected context
            pctx = slim.fully_connected(self.flattened_ctx, 512, activation_fn = None, scope = 'ctx_att')

            rnn_inputs = tf.split(1, self.seq_length + 1, tf.nn.embedding_lookup(self.Wemb, self.labels[:,:self.seq_length + 1]))
            rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]

            prev_h = self.last_hidden_vec(initial_state)

            self.alphas = []
            outputs = []
            for ind in range(self.seq_length + 1):
                if ind > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("attention"):
                    #projected state
                    pstate = slim.fully_connected(prev_h, 512, activation_fn = None, scope = 'h_att')
                    pctx_ = pctx + tf.expand_dims(pstate, 1)
                    pctx_ = tf.nn.tanh(pctx_)
                    alpha = slim.fully_connected(pctx_, 1, activation_fn = None, scope = 'alpha')
                    alpha = tf.squeeze(alpha, [2])
                    alpha = tf.nn.softmax(alpha)
                    self.alphas.append(alpha)
                    weighted_context = tf.reduce_sum(flattened_ctx * tf.expand_dims(alpha, 2), 1)

                cur_output, cur_state = self.cell(tf.concat(1, [weighted_context, rnn_inputs[ind]]), initial_state)
                outputs.append(cur_output)
                prev_h = cur_output

            self.logits = [tf.matmul(output, self.embed_word_W) + self.embed_word_b for output in outputs]

        with tf.variable_scope("loss"):
            loss = tf.nn.seq2seq.sequence_loss_by_example(self.logits,
                    [tf.squeeze(label, [1]) for label in tf.split(1, self.seq_length + 1, self.labels[:, 1:])], # self.labels[:,1:] is the target
                    [tf.squeeze(mask, [1]) for mask in tf.split(1, self.seq_length + 1, self.masks[:, 1:])])
            self.cost = tf.reduce_mean(loss)

        self.final_state = cur_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.cnn_lr = tf.Variable(0.0, trainable=False)

        # Collect the rnn variables, and create the optimizer of rnn
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnnlm')
        grads = utils.clip_by_value(tf.gradients(self.cost, tvars), -self.opt.grad_clip, self.opt.grad_clip)
        #grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
        #        self.opt.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Collect the cnn variables, and create the optimizer of cnn
        cnn_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn')
        cnn_grads = utils.clip_by_value(tf.gradients(self.cost, cnn_tvars), -self.opt.grad_clip, self.opt.grad_clip)
        #cnn_grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, cnn_tvars),
        #        self.opt.grad_clip)
        cnn_optimizer = tf.train.AdamOptimizer(self.cnn_lr)     
        self.cnn_train_op = cnn_optimizer.apply_gradients(zip(cnn_grads, cnn_tvars))

        tf.scalar_summary('training loss', self.cost)
        tf.scalar_summary('learning rate', self.lr)
        tf.scalar_summary('cnn learning rate', self.cnn_lr)
        self.summaries = tf.merge_all_summaries()

    def build_generator(self):
        with tf.variable_scope("rnnlm"):
            #flattened_ctx = tf.reshape(self.context, [self.batch_size, tf.shape(self.context)[1]*tf.shape(self.context)[2],tf.shape(self.context)[3]])
            flattened_ctx = tf.reshape(self.context, [self.batch_size, 196, 512])
            ctx_mean = tf.reduce_mean(flattened_ctx, 1)

            tf.get_variable_scope().reuse_variables()

            initial_state = self.get_initial_state(ctx_mean, self.cell.state_size)

            #projected context
            pctx = slim.fully_connected(flattened_ctx, 512, activation_fn = None, scope = 'ctx_att')

            #rnn_inputs = tf.split(1, self.seq_length + 1, tf.nn.embedding_lookup(self.Wemb, self.labels[:,:self.seq_length + 1]))
            #rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]
            rnn_input = tf.nn.embedding_lookup(self.Wemb, tf.zeros([self.batch_size], tf.int32))

            prev_h = self.last_hidden_vec(initial_state)

            self.g_alphas = []
            outputs = []
            for ind in range(MAX_STEPS):

                with tf.variable_scope("attention"):
                    #projected state
                    pstate = slim.fully_connected(prev_h, 512, activation_fn = None, scope = 'h_att')
                    pctx_ = pctx + tf.expand_dims(pstate, 1)
                    pctx_ = tf.nn.tanh(pctx_)
                    alpha = slim.fully_connected(pctx_, 1, activation_fn = None, scope = 'alpha')
                    alpha = tf.squeeze(alpha, [2])
                    alpha = tf.nn.softmax(alpha)
                    self.g_alphas.append(alpha)
                    weighted_context = tf.reduce_sum(flattened_ctx * tf.expand_dims(alpha, 2), 1)

                cur_output, cur_state = self.cell(tf.concat(1, [weighted_context, rnn_input]), initial_state)
                outputs.append(cur_output)
                prev_h = cur_output

                prev_logit = tf.matmul(prev_h, self.embed_word_W) + self.embed_word_b
                prev_symbol = tf.stop_gradient(tf.argmax(prev_logit, 1))
                rnn_input = tf.nn.embedding_lookup(self.Wemb, prev_symbol)
            
            self.g_output = output = tf.reshape(tf.concat(1, outputs), [-1, self.rnn_size]) # outputs[1:], because we don't calculate loss on time 0.
            self.g_logits = logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b    
            self.g_probs = probs = tf.reshape(tf.nn.softmax(logits), [self.batch_size, MAX_STEPS, self.vocab_size + 1])

        self.generator = tf.argmax(probs, 2)


    def get_placeholder_state(self, state_size, scope = 'placeholder_state'):
        with tf.variable_scope(scope):
            if isinstance(state_size, tf.nn.rnn_cell.LSTMStateTuple):
                c = tf.placeholder(tf.float32, [None, state_size.c], name='LSTM_c')
                h = tf.placeholder(tf.float32, [None, state_size.h], name='LSTM_h')
                return tf.nn.rnn_cell.LSTMStateTuple(c,h)
            elif isinstance(state_size, tuple):
                result = [self.get_placeholder_state(state_size[i], "layer_"+str(i)) for i in xrange(len(state_size))]
                return tuple(result)
            elif isinstance(state_size, int):
                return tf.placeholder(tf.float32, [None, state_size], name='state')

    def flatten_state(self, state):
        if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
            return [state.c, state.h]
        elif isinstance(state, tuple):
            result = []
            for i in xrange(len(state)):
                result += self.flatten_state(state[i])
            return result
        else:
            return [state]

    def build_decoder_rnn(self, first_step):
        with tf.variable_scope("rnnlm"):
            #flattened_ctx = tf.reshape(self.context, [self.batch_size, tf.shape(self.context)[1]*tf.shape(self.context)[2],tf.shape(self.context)[3]])
            flattened_ctx = tf.reshape(self.context, [self.batch_size, 196, 512])
            ctx_mean = tf.reduce_mean(flattened_ctx, 1)

            tf.get_variable_scope().reuse_variables()

            if not first_step:
                initial_state = self.get_placeholder_state(self.cell.state_size)
                self.decoder_flattened_state = self.flatten_state(initial_state)
            else:
                initial_state = self.get_initial_state(ctx_mean, self.cell.state_size)

            self.decoder_prev_word = tf.placeholder(tf.int32, [None])

            if first_step:
                rnn_input = tf.nn.embedding_lookup(self.Wemb, tf.zeros([self.batch_size], tf.int32))
            else:
                rnn_input = tf.nn.embedding_lookup(self.Wemb, self.decoder_prev_word)

            #projected context
            pctx = slim.fully_connected(flattened_ctx, 512, activation_fn = None, scope = 'ctx_att')

            prev_h = self.last_hidden_vec(initial_state)

            alphas = []
            outputs = []

            with tf.variable_scope("attention"):
                #projected state
                pstate = slim.fully_connected(prev_h, 512, activation_fn = None, scope = 'h_att')
                pctx_ = pctx + tf.expand_dims(pstate, 1)
                pctx_ = tf.nn.tanh(pctx_)
                alpha = slim.fully_connected(pctx_, 1, activation_fn = None, scope = 'alpha')
                alpha = tf.squeeze(alpha, [2])
                alpha = tf.nn.softmax(alpha)
                alphas.append(alpha)
                weighted_context = tf.reduce_sum(flattened_ctx * tf.expand_dims(alpha, 2), 1)

            output, state = self.cell(tf.concat(1, [weighted_context, rnn_input]), initial_state)
            logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b
            decoder_probs = tf.reshape(tf.nn.softmax(logits), [self.batch_size, self.vocab_size + 1])
            decoder_state = self.flatten_state(state)
        return [decoder_probs, decoder_state]


    def build_decoder(self):
        self.decoder_model_init = self.build_decoder_rnn(True)
        self.decoder_model_cont = self.build_decoder_rnn(False)

    def decode(self, img, beam_size, sess, max_steps=30):
        """Decode an image with a sentences."""
        
        # Initilize beam search variables
        # Candidate will be represented with a dictionary
        #   "indexes": a list with indexes denoted a sentence; 
        #   "words": word in the decoded sentence without <bos>
        #   "score": log-likelihood of the sentence
        #   "state": RNN state when generating the last word of the candidate
        good_sentences = [] # store sentences already ended with <bos>
        cur_best_cand = [] # store current best candidates
        highest_score = 0.0 # hightest log-likelihodd in good sentences
        
        # Get the initial logit and state
        cand = {'indexes': [], 'score': 0}
        cur_best_cand.append(cand)
            
        # Expand the current best candidates until max_steps or no candidate
        for i in xrange(max_steps + 1):
            # expand candidates
            cand_pool = []
            #for cand in cur_best_cand:
                #probs, state = self.get_probs_cont(cand['state'], cand['indexes'][-1], sess)
            if i == 0:
                all_probs, all_states = self.get_probs_init(img, sess)
            else:
                states = [np.vstack([cand['state'][i] for cand in cur_best_cand]) for i in xrange(len(cur_best_cand[0]['state']))]
                indexes = [cand['indexes'][-1] for cand in cur_best_cand]
                imgs = np.vstack([img] * len(cur_best_cand))
                all_probs, all_states = self.get_probs_cont(states, imgs, indexes, sess)
            for ind_cand in range(len(cur_best_cand)):
                cand = cur_best_cand[ind_cand]
                probs = all_probs[ind_cand]
                state = [x[ind_cand] for x in all_states]
                
                probs = np.squeeze(probs)
                probs_order = np.argsort(-probs)
                for ind_b in xrange(beam_size):
                    cand_e = copy.deepcopy(cand)
                    cand_e['indexes'].append(probs_order[ind_b])
                    cand_e['score'] -= np.log(probs[probs_order[ind_b]])
                    cand_e['state'] = state
                    cand_pool.append(cand_e)
            # get final cand_pool
            cur_best_cand = sorted(cand_pool, key=lambda cand: cand['score'])
            cur_best_cand = self.truncate_list(cur_best_cand, beam_size)

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
        good_sentences = self.truncate_list(good_sentences, beam_size)

        return [sent['indexes'] for sent in good_sentences]
        
    def truncate_list(self, l, num):
        if num == -1:
            num = len(l)
        return l[:min(len(l), num)]

    def get_probs_init(self, img, sess):
        """Use the model to get initial logit"""
        m = self.decoder_model_init
        
        probs, state = sess.run(m, {self.images: img})
                                                            
        return (probs, state)
        
    def get_probs_cont(self, prev_state, img, prev_word, sess):
        """Use the model to get continued logit"""
        m = self.decoder_model_cont
        prev_word = np.array(prev_word, dtype='int32')

        pointer = [self.images, self.decoder_prev_word] + self.decoder_flattened_state
        feeded = [img, prev_word] + prev_state
        
        probs, state = sess.run(m, {pointer[i]: feeded[i] for i in xrange(len(pointer))})
                                                            
        return (probs, state)