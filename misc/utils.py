from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections
import six

# My own clip by value which could input a list of tensors
def clip_by_value(t_list, clip_value_min, clip_value_max, name=None):
    if (not isinstance(t_list, collections.Sequence)
            or isinstance(t_list, six.string_types)):
        raise TypeError("t_list should be a sequence")
    t_list = list(t_list)
        
    with tf.name_scope(name or "clip_by_value") as name:
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

# Truncate the list of beam given a maximum length
def truncate_list(l, max_len):
    if max_len == -1:
        max_len = len(l)
    return l[:min(len(l),  max_len)]

# Turn nested state into a flattened list
# Used both for flattening the nested placeholder states and for output states value of previous time step
def flatten_state(state):
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        return [state.c, state.h]
    elif isinstance(state, tuple):
        result = []
        for i in xrange(len(state)):
            result += flatten_state(state[i])
        return result
    else:
        return [state]

# When decoding step by step: we need to initialize the state of next timestep according to the previous time step.
# Because states could be nested tuples or lists, so we get the states recursively.
def get_placeholder_state(state_size, scope = 'placeholder_state'):
    with tf.variable_scope(scope):
        if isinstance(state_size, tf.contrib.rnn.LSTMStateTuple):
            c = tf.placeholder(tf.float32, [None, state_size.c], name='LSTM_c')
            h = tf.placeholder(tf.float32, [None, state_size.h], name='LSTM_h')
            return tf.contrib.rnn.LSTMStateTuple(c,h)
        elif isinstance(state_size, tuple):
            result = [get_placeholder_state(state_size[i], "layer_"+str(i)) for i in xrange(len(state_size))]
            return tuple(result)
        elif isinstance(state_size, int):
            return tf.placeholder(tf.float32, [None, state_size], name='state')

# Get the last hidden vector. (The hidden vector of the deepest layer)
# For the input of the attention model of next time step.
def last_hidden_vec(state):
    if isinstance(state, tuple):
        return last_hidden_vec(state[len(state) - 1])
    elif isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        return state.h
    else:
        return state

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.shape
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def get_initial_state(input, state_size, scope = 'init_state'):
    """
    Recursively initialize the first state.

    state_size is a nested of tuple and LSTMStateTuple and integer.
        
    It is so complicated because we use state_is_tuple
    """

    with tf.variable_scope(scope):
        if isinstance(state_size, tf.contrib.rnn.LSTMStateTuple):
            c = slim.fully_connected(input, state_size.c, activation_fn=tf.nn.tanh, scope='LSTM_c')
            h = slim.fully_connected(input, state_size.h, activation_fn=tf.nn.tanh, scope='LSTM_h')
            return tf.contrib.rnn.LSTMStateTuple(c,h)
        elif isinstance(state_size, tuple):
            result = [get_initial_state(input, state_size[i], "layer_"+str(i)) for i in xrange(len(state_size))]
            return tuple(result)
        elif isinstance(state_size, int):
            return slim.fully_connected(input, state_size, activation_fn=tf.nn.tanh, scope='state')

def expand_feat(input, multiples, scope = 'expand_feat'):
    """
    Expand the dimension of states;
    According to multiples.

    Similar reason why it's so complicated.
    """
    with tf.variable_scope(scope):
        if isinstance(input, tf.contrib.rnn.LSTMStateTuple):
            c = expand_feat(input.c, multiples, scope='expand_LSTM_c')
            h = expand_feat(input.h, multiples, scope='expand_LSTM_c')
            return tf.contrib.rnn.LSTMStateTuple(c,h)
        elif isinstance(input, tuple):
            result = [expand_feat(input[i], multiples, "expand_layer_"+str(i)) for i in xrange(len(input))]
            return tuple(result)
        else:
            return tf.reshape(tf.tile(tf.expand_dims(input, 1), [1, multiples, 1]), [tf.shape(input)[0] * multiples, input.get_shape()[1].value])

def get_optimizer(opt, lr):
    if opt.optim == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr, momentum=opt.optim_alpha, epsilon=opt.optim_epsilon)
    elif opt.optim == 'adagrad':
        return tf.train.AdagradOptimizer(lr)
    elif opt.optim == 'sgd':
        return tf.train.GradientDescentOptimizer(lr)
    elif opt.optim == 'sgdm':
        return tf.train.MomentumOptimizer(lr, opt.optim_alpha)
    elif opt.optim == 'sgdmom':
        return tf.train.MomentumOptimizer(lr, opt.optim_alpha, use_nesterov=True)
    elif opt.optim == 'adam':
        return tf.train.AdamOptimizer(lr, beta1=opt.optim_alpha, beta2=opt.optim_beta, epsilon=opt.optim_epsilon)
    else:
        raise Exception('bad option opt.optim')

def get_cnn_optimizer(opt, cnn_lr):
    if opt.cnn_optim == 'rmsprop':
        return tf.train.RMSPropOptimizer(cnn_lr, momentum=opt.cnn_optim_alpha, epsilon=opt.optim_epsilon)
    elif opt.cnn_optim == 'adagrad':
        return tf.train.AdagradOptimizer(cnn_lr)
    elif opt.cnn_optim == 'sgd':
        return tf.train.GradientDescentOptimizer(cnn_lr)
    elif opt.cnn_optim == 'sgdm':
        return tf.train.MomentumOptimizer(cnn_lr, opt.cnn_optim_alpha)
    elif opt.cnn_optim == 'sgdmom':
        return tf.train.MomentumOptimizer(cnn_lr, opt.cnn_optim_alpha, use_nesterov=True)
    elif opt.cnn_optim == 'adam':
        return tf.train.AdamOptimizer(cnn_lr, beta1=opt.cnn_optim_alpha, beta2=opt.cnn_optim_beta, epsilon=opt.optim_epsilon)
    else:
        raise Exception('bad option opt.cnn_optim')
