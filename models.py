import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
import os
import vgg
import copy

import numpy as np
import collections
import six

def setup(opt):
    
    # check compatibility if training is continued from previously saved model
    if opt.start_from is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        ckpt = tf.train.get_checkpoint_state(opt.start_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"
        opt.ckpt = ckpt

    return Model(opt)

# My own clip by value which could input a list of tensors
def clip_by_value(t_list, clip_value_min, clip_value_max, name=None):
    if (not isinstance(t_list, collections.Sequence)
            or isinstance(t_list, six.string_types)):
        raise TypeError("t_list should be a sequence")
    t_list = list(t_list)
        
    with tf.name_scope(name or "clip_by_value") as name:
    #with ops.name_scope(name, "clip_by_global_norm",
    #              t_list + [clip_norm]) as name:
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        values = [
            tf.convert_to_tensor(
                t.values if isinstance(t, tf.IndexedSlices) else t,
                name="t_%d" % i)
            if t is not None else t
            for i, t in enumerate(t_list)]
        values_clipped = []
        for i, v in enumerate(values):
            if v is None:
                values_clipped.append(None)
            else:
                with tf.get_default_graph().colocate_with(v):
                    values_clipped.append(
                        tf.clip_by_value(v, clip_value_min, clip_value_max))

        list_clipped = [
            tf.IndexedSlices(c_v, t.indices, t.dense_shape)
            if isinstance(t, tf.IndexedSlices)
            else c_v
            for (c_v, t) in zip(values_clipped, t_list)]

    return list_clipped

class Model():
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
        self.fc7 = self.cnn.drop7
        self.cnn_training = self.cnn.training
        """
        # Old model loading
        with open(self.opt.cnn_model) as f:
            fileContent = f.read()
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fileContent)
            tf.import_graph_def(graph_def, input_map={"images": self.images}, name='vgg16')
            self.vgg16 = tf.get_default_graph()

        self.fc7 = self.vgg16.get_tensor_by_name("vgg16/Relu_1:0")
        """

        # Variable in language model
        with tf.variable_scope("rnnlm"):
            # Word Embedding table
            #with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([self.vocab_size + 1, self.input_encoding_size], -0.1, 0.1), name='Wemb')
            self.bemb = self.init_bias(self.input_encoding_size, name='bemb')

            # Image Emcoding
            self.encode_img_W = tf.Variable(tf.random_uniform([4096, self.input_encoding_size], -0.1, 0.1), name='encode_img_W')
            self.encode_img_b = self.init_bias(self.input_encoding_size, name='encode_img_b')

            #
            self.embed_word_W = tf.Variable(tf.random_uniform([self.rnn_size, self.vocab_size + 1], -0.1, 0.1), name='embed_word_W')
            self.embed_word_b = self.init_bias(self.vocab_size + 1, name='embed_word_b')

            # RNN cell
            if opt.rnn_type == 'rnn':
                self.cell_fn = cell_fn = rnn_cell.BasicRNNCell
            elif opt.rnn_type == 'gru':
                self.cell_fn = cell_fn = rnn_cell.GRUCell
            elif opt.rnn_type == 'lstm':
                self.cell_fn = cell_fn = rnn_cell.BasicLSTMCell
            else:
                raise Exception("RNN type not supported: {}".format(opt.rnn_type))

            self.keep_prob = tf.cond(self.training, 
                                lambda : tf.constant(1 - self.drop_prob_lm),
                                lambda : tf.constant(1.0), name = 'keep_prob')

            self.basic_cell = cell = rnn_cell.DropoutWrapper(cell_fn(self.rnn_size, state_is_tuple = True), 1.0, self.keep_prob)

            self.cell = rnn_cell.MultiRNNCell([cell] * opt.num_layers, state_is_tuple = True)

    def build_model(self):
        with tf.name_scope("batch_size"):
            self.batch_size = tf.shape(self.images)[0]
        with tf.variable_scope("rnnlm"):
            image_emb = tf.matmul(self.fc7, self.encode_img_W) + self.encode_img_b

            # Replicate self.seq_per_img times for each image embedding
            image_emb = tf.reshape(tf.tile(tf.expand_dims(image_emb, 1), [1, self.seq_per_img, 1]), [self.batch_size * self.seq_per_img, self.input_encoding_size])

            rnn_inputs = tf.split(1, self.seq_length + 1, tf.nn.embedding_lookup(self.Wemb, self.labels[:,:self.seq_length + 1]) + self.bemb)
            rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]
            rnn_inputs = [image_emb] + rnn_inputs

            initial_state = self.cell.zero_state(self.batch_size * self.seq_per_img, tf.float32)

            outputs, last_state = seq2seq.rnn_decoder(rnn_inputs, initial_state, self.cell, loop_function=None)
            #outputs, last_state = tf.nn.rnn(self.cell, rnn_inputs, initial_state)

            self.logits = [tf.matmul(output, self.embed_word_W) + self.embed_word_b for output in outputs[1:]]
        with tf.variable_scope("loss"):
            loss = seq2seq.sequence_loss_by_example(self.logits,
                    [tf.squeeze(label, [1]) for label in tf.split(1, self.seq_length + 1, self.labels[:, 1:])], # self.labels[:,1:] is the target
                    [tf.squeeze(mask, [1]) for mask in tf.split(1, self.seq_length + 1, self.masks[:, 1:])])
            self.cost = tf.reduce_mean(loss)
        
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.cnn_lr = tf.Variable(0.0, trainable=False)

        # Collect the rnn variables, and create the optimizer of rnn
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnnlm')
        grads = clip_by_value(tf.gradients(self.cost, tvars), -self.opt.grad_clip, self.opt.grad_clip)
        #grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
        #        self.opt.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Collect the cnn variables, and create the optimizer of cnn
        cnn_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn')
        cnn_grads = clip_by_value(tf.gradients(self.cost, cnn_tvars), -self.opt.grad_clip, self.opt.grad_clip)
        #cnn_grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, cnn_tvars),
        #        self.opt.grad_clip)
        cnn_optimizer = tf.train.AdamOptimizer(self.cnn_lr)     
        self.cnn_train_op = optimizer.apply_gradients(zip(cnn_grads, cnn_tvars))

        tf.scalar_summary('training loss', self.cost)
        tf.scalar_summary('learning rate', self.lr)
        tf.scalar_summary('cnn learning rate', self.cnn_lr)
        self.summaries = tf.merge_all_summaries()

    def build_generator(self):
        with tf.variable_scope("rnnlm"):
            image_emb = tf.matmul(self.fc7, self.encode_img_W) + self.encode_img_b

            rnn_inputs = tf.split(1, self.seq_length + 1, tf.zeros([self.batch_size, self.seq_length + 1, self.input_encoding_size]))
            rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]
            rnn_inputs = [image_emb] + rnn_inputs

            initial_state = self.cell.zero_state(self.batch_size, tf.float32)

            # Always pick the word with largest probability as the input of next time step
            def loop(prev, i):
                if i == 1:
                    return rnn_inputs[1]
                prev = tf.matmul(prev, self.embed_word_W) + self.embed_word_b
                prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                return tf.nn.embedding_lookup(self.Wemb, prev_symbol) + self.bemb

            tf.get_variable_scope().reuse_variables()
            outputs, last_state = seq2seq.rnn_decoder(rnn_inputs, initial_state, self.cell, loop_function=loop)
            #outputs, last_state = tf.nn.rnn(self.cell, rnn_inputs, initial_state)
            self.g_output = output = tf.reshape(tf.concat(1, outputs[1:]), [-1, self.rnn_size]) # outputs[1:], because we don't calculate loss on time 0.
            self.g_logits = logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b
            self.g_probs = probs = tf.reshape(tf.nn.softmax(logits), [self.batch_size, self.seq_length + 1, self.vocab_size + 1])

        self.generator = tf.argmax(probs, 2)

    def build_decoder_rnn(self, first_step):
        with tf.variable_scope("rnnlm"):
            if first_step:
                rnn_input = tf.matmul(self.fc7, self.encode_img_W) + self.encode_img_b
            else:
                self.decoder_prev_word = tf.placeholder(tf.int32, [None])
                rnn_input = tf.nn.embedding_lookup(self.Wemb, self.decoder_prev_word) + self.bemb

            self.batch_size = tf.shape(rnn_input)[0]

            tf.get_variable_scope().reuse_variables()
            basic_cell = rnn_cell.DropoutWrapper(self.cell_fn(self.rnn_size, state_is_tuple = False), 1.0, self.keep_prob)
            self.decoder_cell = rnn_cell.MultiRNNCell([basic_cell] * self.opt.num_layers, state_is_tuple = False)
            state_size = self.decoder_cell.state_size
            if not first_step:
                self.decoder_initial_state = initial_state = tf.placeholder(tf.float32, 
                    [None, state_size])
            else:
                initial_state = self.decoder_cell.zero_state(
                    self.batch_size, tf.float32)

            outputs, state = seq2seq.rnn_decoder([rnn_input], initial_state, self.decoder_cell)
            #outputs, state = tf.nn.rnn(self.decoder_cell, [rnn_input], initial_state)
            logits = tf.matmul(outputs[0], self.embed_word_W) + self.embed_word_b
            decoder_probs = tf.reshape(tf.nn.softmax(logits), [self.batch_size, self.vocab_size + 1])
            decoder_state = state
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
        probs_init, state_init = self.get_probs_init(img, sess)
        cand = {'indexes': [0], 'score': 0, 'state': state_init}
        cur_best_cand.append(cand)
            
        # Expand the current best candidates until max_steps or no candidate
        for i in xrange(max_steps):
            # expand candidates
            cand_pool = []
            #for cand in cur_best_cand:
                #probs, state = self.get_probs_cont(cand['state'], cand['indexes'][-1], sess)
            states = np.vstack([cand['state'] for cand in cur_best_cand])
            indexes = [cand['indexes'][-1] for cand in cur_best_cand]
            all_probs, all_states = self.get_probs_cont(states, indexes, sess)
            for ind_cand in range(len(cur_best_cand)):
                cand = cur_best_cand[ind_cand]
                probs = all_probs[ind_cand]
                state = all_states[ind_cand]
                
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
            if cand['indexes'][-1] != vocab['<bos>']:
                cand['indexes'].append(vocab['<bos>'])
            good_sentences.append(cand)
            highest_score = max(highest_score, cand['score'])
            
        # Sort good sentences and return the final list
        good_sentences = sorted(good_sentences, key=lambda cand: cand['score'])
        good_sentences = self.truncate_list(good_sentences, beam_size)

        
        return [sent['indexes'][1:] for sent in good_sentences]
        
    def truncate_list(self, l, num):
        if num == -1:
            num = len(l)
        return l[:min(len(l), num)]

    def get_probs_init(self, img, sess):
        """Use the model to get initial logit"""
        m = self.decoder_model_init
        
        probs, state = sess.run(m, {self.images: img})
                                                            
        return (probs, state)
        
    def get_probs_cont(self, state_prev, prev_word, sess):
        """Use the model to get continued logit"""
        m = self.decoder_model_cont
        prev_word = np.array(prev_word, dtype='int32')
        
        probs, state = sess.run(m,{self.decoder_prev_word: prev_word,
                                         self.decoder_initial_state: state_prev})
                                                            
        return (probs, state)
