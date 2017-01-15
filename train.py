from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils

import os
NUM_THREADS = 2 #int(os.environ['OMP_NUM_THREADS'])

#from ipdb import set_trace

def train(opt):
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    model = models.setup(opt)

    infos = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model.build_model()
    model.build_generator()
    model.build_decoder()

    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads=NUM_THREADS
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        # Initialize the variables, and restore the variables form checkpoint if there is.
        # and initialize the writer
        model.initialize(sess)
        
        # Assign the learning rate
        if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
            frac = (epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
            decay_factor = 0.5  ** frac
            sess.run(tf.assign(model.lr, opt.learning_rate * decay_factor)) # set the decayed rate
            sess.run(tf.assign(model.cnn_lr, opt.cnn_learning_rate * decay_factor))
        else:
            sess.run(tf.assign(model.lr, opt.learning_rate))
            sess.run(tf.assign(model.cnn_lr, opt.cnn_learning_rate))
        # Assure in training mode
        sess.run(tf.assign(model.training, True))
        sess.run(tf.assign(model.cnn_training, True))

        while True:
            start = time.time()
            # Load data from train split (0)
            data = loader.get_batch('train')
            print('Read data:', time.time() - start)

            start = time.time()
            feed = {model.images: data['images'], model.labels: data['labels'], model.masks: data['masks']}
            if iteration <= opt.finetune_cnn_after or opt.finetune_cnn_after == -1:
                train_loss, merged, _ = sess.run([model.cost, model.summaries, model.train_op], feed)
            else:
                # Finetune the cnn
                train_loss, merged, _, __ = sess.run([model.cost, model.summaries, model.train_op, model.cnn_train_op], feed)
            end = time.time()
            print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, end - start))

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1

            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                model.summary_writer.add_summary(merged, iteration)
                model.summary_writer.flush()
                loss_history[iteration] = train_loss

            # make evaluation on validation set, and save model
            if (iteration % opt.save_checkpoint_every == 0):
                # eval model
                eval_kwargs = {'val_images_use': opt.val_images_use,
                                'split': 'val',
                                'language_eval': opt.language_eval, 
                                'dataset': opt.input_json}
                val_loss, predictions, lang_stats = eval_split(sess, model, loader, eval_kwargs)

                # Write validation result into summary
                summary = tf.Summary(value=[tf.Summary.Value(tag='validation loss', simple_value=val_loss)])
                model.summary_writer.add_summary(summary, iteration)
                for k,v in lang_stats.iteritems():
                    summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v)])
                    model.summary_writer.add_summary(summary, iteration)
                model.summary_writer.flush()
                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                if best_val_score is None or current_score > best_val_score: # if true
                    best_val_score = current_score
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model.ckpt')
                    model.saver.save(sess, checkpoint_path, global_step = iteration)
                    print("model saved to {}".format(checkpoint_path))

                    # Dump miscalleous informations
                    infos['iter'] = iteration
                    infos['epoch'] = epoch
                    infos['iterators'] = loader.iterators
                    infos['best_val_score'] = best_val_score
                    infos['opt'] = opt
                    infos['val_result_history'] = val_result_history
                    infos['loss_history'] = loss_history
                    infos['vocab'] = loader.get_vocab()
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break

def eval_split(sess, model, loader, eval_kwargs):
    verbose = eval_kwargs.get('verbose', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    language_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    sess.run(tf.assign(model.training, False))
    sess.run(tf.assign(model.cnn_training, False))

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 0
    predictions = []
    while True:
        if opt.beam_size > 1:
            data = loader.get_batch(split, 1)
            n = n + 1
        else:
            data = loader.get_batch(split)
            n = n + loader.batch_size

        # forward the model to get loss
        feed = {model.images: data['images'], model.labels: data['labels'], model.masks: data['masks']}
        loss = sess.run(model.cost, feed)

        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        if opt.beam_size == 1:
            # forward the model to also get generated samples for each image
            feed = {model.images: data['images']}
            #g_o,g_l,g_p, seq = sess.run([model.g_output, model.g_logits, model.g_probs, model.generator], feed)
            seq = sess.run(model.generator, feed)

            #set_trace()
            sents = utils.decode_sequence(loader.get_vocab(), seq)

            for k, sent in enumerate(sents):
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                predictions.append(entry)
                if verbose:
                    print('image %s: %s' %(entry['image_id'], entry['caption']))
        else:
            seq = model.decode(data['images'], opt.beam_size, sess)
            sents = [' '.join([loader.ix_to_word.get(str(ix), '') for ix in sent]).strip() for sent in seq]
            entry = {'image_id': data['infos'][0]['id'], 'caption': sents[0]}
            predictions.append(entry)
            if verbose:
                for sent in sents:
                    print('image %s: %s' %(entry['image_id'], sent))
        
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(n - ix1):
            predictions.pop()
        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if n>= val_images_use:
            break

    if language_eval == 1:
        lang_stats = eval_utils.language_eval(dataset, predictions)

    # Switch back to training mode
    sess.run(tf.assign(model.training, True))
    sess.run(tf.assign(model.cnn_training, True))
    return loss_sum/loss_evals, predictions, lang_stats

opt = opts.parse_opt()
train(opt)
