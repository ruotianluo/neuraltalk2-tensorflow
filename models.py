import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
import os

import numpy as np

def setup(opt):
    
    # check compatibility if training is continued from previously saved model
    if opt.start_from is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % start.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos.pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        ckpt = tf.train.get_checkpoint_state(opt.start_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"
        opt.ckpt = ckpt

    return Model(opt)


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
        self.dropout = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.vocab_size = opt.vocab_size
        self.seq_per_img = opt.seq_per_img
        self.batch_size = opt.batch_size

        self.opt = opt

        # Variable indicating in training mode or evaluation mode
        self.training = tf.Variable(True, trainable = False, name = "training")

        # Input varaibles
        self.images = tf.placeholder(tf.float32, [self.batch_size, 224, 224, 3], name = "images")
        self.labels = tf.placeholder(tf.int32, [self.batch_size * self.seq_per_img, self.seq_length + 2], name = "labels")
        self.masks = tf.placeholder(tf.float32, [self.batch_size * self.seq_per_img, self.seq_length + 2], name = "masks")

        # VGG 16
        
        with open(self.opt.cnn_model) as f:
            fileContent = f.read()
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fileContent)
            tf.import_graph_def(graph_def, input_map={"images": self.images}, name='vgg16')
            self.vgg16 = tf.get_default_graph()

        self.fc7 = self.vgg16.get_tensor_by_name("vgg16/Relu_1:0")

        # Variable in language model
        with tf.variable_scope("rnnlm"):
            # Word Embedding table
            with tf.device("/cpu:0"):
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
            cell_fn = rnn_cell.BasicRNNCell
        elif opt.rnn_type == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif opt.rnn_type == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("RNN type not supported: {}".format(opt.rnn_type))

        self.keep_prob = tf.cond(self.training, 
                            lambda : tf.constant(1 - opt.drop_prob_lm),
                            lambda : tf.constant(1.0))

        cell = rnn_cell.DropoutWrapper(cell_fn(self.rnn_size, state_is_tuple = True), 1.0, self.keep_prob)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * opt.num_layers, state_is_tuple = True)

    def build_model(self):
        image_emb = tf.matmul(self.fc7, self.encode_img_W) + self.encode_img_b

        # Replicate self.seq_per_img times for each image embedding
        image_emb = tf.reshape(tf.tile(tf.expand_dims(image_emb, 1), [1, self.seq_per_img, 1]), [self.batch_size * self.seq_per_img, self.input_encoding_size])

        rnn_inputs = tf.split(1, self.seq_length + 1, tf.nn.embedding_lookup(self.Wemb, self.labels[:,:self.seq_length + 1]) + self.bemb)
        rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]
        rnn_inputs = [image_emb] + rnn_inputs

        initial_state = self.cell.zero_state(self.batch_size * self.seq_per_img, tf.float32)

        outputs, last_state = seq2seq.rnn_decoder(rnn_inputs, initial_state, self.cell, loop_function=None, scope='rnnlm')
        """
        output = tf.reshape(tf.concat(1, outputs[1:]), [-1, self.rnn_size]) # outputs[1:], because we don't calculate loss on time 0.
        self.logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.labels[:, 1:], [-1])], # self.labels[:,1:] is the target
                [tf.reshape(self.masks[:, 1:], [-1])])
        self.cost = tf.reduce_sum(loss)/self.batch_size/self.seq_per_img
        """
        self.logits = [tf.matmul(output, self.embed_word_W) + self.embed_word_b for output in outputs[1:]]
        loss = seq2seq.sequence_loss_by_example(self.logits,
                [tf.squeeze(label, [1]) for label in tf.split(1, self.seq_length + 1, self.labels[:, 1:])], # self.labels[:,1:] is the target
                [tf.squeeze(mask, [1]) for mask in tf.split(1, self.seq_length + 1, self.masks[:, 1:])])
        self.cost = tf.reduce_mean(loss)
        
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.cnn_lr = tf.Variable(0.0, trainable=False)

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnnlm')
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                self.opt.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        """
        cnn_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg16')
        cnn_grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, cnn_tvars),
                self.opt.grad_clip)
        cnn_optimizer = tf.train.AdamOptimizer(self.cnn_lr)     
        self.cnn_train_op = optimizer.apply_gradients(zip(cnn_grads, cnn_tvars))
        """

        tf.scalar_summary('training loss', self.cost)
        tf.scalar_summary('learning rate', self.lr)
        tf.scalar_summary('cnn learning rate', self.cnn_lr)
        self.summaries = tf.merge_all_summaries()

    def build_generator(self):
        image_emb = tf.matmul(self.fc7, self.encode_img_W) + self.encode_img_b

        rnn_inputs = tf.split(1, self.seq_length + 1, tf.zeros([self.batch_size, self.seq_length + 1, self.input_encoding_size]))
        rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]
        rnn_inputs = [image_emb] + rnn_inputs

        initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        def loop(prev, i):
            if i == 1:
                return rnn_inputs[1]
            prev = tf.matmul(prev, self.embed_word_W) + self.embed_word_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(self.Wemb, prev_symbol) + self.bemb

        tf.get_variable_scope().reuse_variables()

        outputs, last_state = seq2seq.rnn_decoder(rnn_inputs, initial_state, self.cell, loop_function=loop, scope='rnnlm')
        self.g_output = output = tf.reshape(tf.concat(1, outputs[1:]), [-1, self.rnn_size]) # outputs[1:], because we don't calculate loss on time 0.
        self.g_logits = logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b
        self.g_probs = probs = tf.reshape(tf.nn.softmax(logits), [self.batch_size, self.seq_length + 1, self.vocab_size + 1])
        self.generator = tf.argmax(probs, 2)
