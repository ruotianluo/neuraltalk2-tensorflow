from simpleloader import *
import tensorflow as tf

import opts

opt = opts.parse_opt()
loader = DataLoader(opt)
sess = tf.InteractiveSession()
loader.assign_session(sess)

count = 0
start = time.time()
while True:
	data = loader.get_batch(0)
	count += 1
	if data['bounds']['wrapped']:
		break
end = time.time()
print 'Time in total:', end-start
print 'Total batch number:', count
print 'Average time:', (end-start)/count


