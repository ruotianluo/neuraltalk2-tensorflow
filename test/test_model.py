import tensorflow as tf
import models
import opts
import numpy as np


opt = opts.parse_opt()
opt.batch_size = 2
opt.seq_length = 5
opt.seq_per_img = 2
sess = tf.InteractiveSession()

data = {}
im1 = np.random.random([1,224,224,3])
data['images'] = np.vstack([im1, -im1])
data['labels'] = np.array([[0,1,2,3,4,0,0],[0,6,7,8,9,10,0],[0,1,2,3,4,0,0],[0,6,7,8,9,10,0]])
data['masks'] = np.array([[0,1,1,1,1,0,0],[0,1,1,1,1,1,0],[0,1,1,1,1,0,0],[0,1,1,1,1,1,0]])

opt.vocab_size = 10
model = models.Model(opt)

model.build_model()
model.build_generator()
tf.global_variables_initializer().run()
sess.run(tf.assign(model.lr, 0.01))
feed = {model.images: data['images'], model.labels: data['labels'], model.masks: data['masks'], model.keep_prob: 1.0}
train_loss, _ = sess.run([model.cost, model.train_op], feed)

seq = sess.run(model.generator, feed)
print(seq)