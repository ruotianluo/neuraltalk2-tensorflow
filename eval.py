from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import tensorflow as tf

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils

NUM_THREADS = 2 #int(os.environ['OMP_NUM_THREADS'])

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=2,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', 
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='', 
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_h5', type=str, default='', 
                help='path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
parser.add_argument('--input_json', type=str, default='', 
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test', 
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', 
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='evalscript', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')

opt = parser.parse_args()

# Load infos
with open(opt.infos_path) as f:
    infos = cPickle.load(f)

# override and collect parameters
if len(opt.input_h5) == 0:
    opt.input_h5 = infos['opt'].input_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
ignore = ["id", "batch_size", "beam_size", "start_from"]
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
model = models.setup(opt)
model.build_model()
model.build_generator()
model.build_decoder()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size})

# Evaluation fun(ction)
def eval_split(sess, model, loader, eval_kwargs):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', -1)
    split = eval_kwargs.get('split', 'test')
    language_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    sess.run(tf.assign(model.training, False))
    sess.run(tf.assign(model.cnn_training, False))

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []

    while True:
        # fetch a batch of data
        if opt.beam_size > 1:
            data = loader.get_batch(split, 1)
            n = n + 1
        else:
            data = loader.get_batch(split, opt.batch_size)
            n = n + opt.batch_size

        #evaluate loss if we have the labels
        loss = 0
        if data.get('labels', None) is not None:
            # forward the model to get loss
            feed = {model.images: data['images'], model.labels: data['labels'], model.masks: data['masks']}
            loss = sess.run(model.cost, feed)
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        if opt.beam_size == 1:
            # forward the model to also get generated samples for each image
            feed = {model.images: data['images']}
            #g_o,g_l,g_p, seq = sess.run([model.g_output, model.g_logits, model.g_probs, model.generator], feed)
            seq = sess.run(model.generator, feed)

            #set_trace()
            sents = utils.decode_sequence(vocab, seq)

            for k, sent in enumerate(sents):
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                predictions.append(entry)
                if verbose:
                    print('image %s: %s' %(entry['image_id'], entry['caption']))
        else:
            seq = model.decode(data['images'], opt.beam_size, sess)
            sents = [' '.join([vocab.get(str(ix), '') for ix in sent]).strip() for sent in seq]
            sents = [sents[0]]
            entry = {'image_id': data['infos'][0]['id'], 'caption': sents[0]}
            predictions.append(entry)
            if verbose:
                for sent in sents:
                    print('image %s: %s' %(entry['image_id'], sent))

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if opt.dump_path == 1:
                entry['file_name'] = data['infos'][k]['file_path']
                table.insert(predictions, entry)
            if opt.dump_images == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(opt.image_root, data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if language_eval == 1:
        lang_stats = eval_utils.language_eval(dataset, predictions)

    # Switch back to training mode
    sess.run(tf.assign(model.training, True))
    sess.run(tf.assign(model.cnn_training, True))
    return loss_sum/loss_evals, predictions, lang_stats

tf_config = tf.ConfigProto()
tf_config.intra_op_parallelism_threads=NUM_THREADS
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    # Initilize the variables
    sess.run(tf.global_variables_initializer())
    # Load the model checkpoint to evaluate
    assert len(opt.model) > 0, 'must provide a model'
    tf.train.Saver(tf.trainable_variables()).restore(sess, opt.model)

    # Set sample options
    sess.run(tf.assign(model.sample_max, opt.sample_max == 1))
    sess.run(tf.assign(model.sample_temperature, opt.temperature))

    loss, split_predictions, lang_stats = eval_split(sess, model, loader, 
        {'num_images': opt.num_images,
        'language_eval': opt.language_eval,
        'split': opt.split})

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
