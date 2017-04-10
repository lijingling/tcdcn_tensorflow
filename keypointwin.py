import tensorflow as tf
from PrepareData import get_batch
from PrepareData import get_testset
import numpy as np 

# Parameters
isTrain = False
starter_learning_rate = 0.001
lr_iter = 800
lr_base = 0.9
training_iters = 20000
batch_size = 64
testbatch = 5
display_step = 5

checkpoint_steps = 100  
checkpoint_dir = 'tfresult/'

# Network Parameters
landmark_num = 68 * 2
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
image = tf.placeholder(tf.float32, shape=[None, 40, 40])
landmark = tf.placeholder(tf.float32, shape=[None, landmark_num])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# Create model
def conv_net(image, weights, biases, dropout):
	# Reshape input picture
	x_image = tf.reshape(image, [-1, 40, 40, 1])
	# Layer1
	h_conv1 = tf.abs(tf.nn.tanh(conv2d(x_image, weights['wc1']) + biases['bc1']))
	h_pool1 = max_pool_2x2(h_conv1)

	# Layer2
	h_conv2 = tf.abs(tf.nn.tanh(conv2d(h_pool1, weights['wc2']) + biases['bc2']))
	h_pool2 = max_pool_2x2(h_conv2)

	# Layer3
	h_conv3 = tf.abs(tf.nn.tanh(conv2d(h_pool2, weights['wc3']) + biases['bc3']))
	h_pool3 = max_pool_2x2(h_conv3)

	# Layer4
	h_conv4 = tf.abs(tf.nn.tanh(conv2d(h_pool3, weights['wc4']) + biases['bc4']))
	h_pool4 = h_conv4

	# Fully connected layer
	# Reshape conv4 output to fit fully connected layer input
	h_pool4_flat = tf.reshape(h_pool4, [-1, 2 * 2 * 64])
	h_fc1 = tf.abs(tf.nn.tanh(tf.matmul(h_pool4_flat, weights['wd1']) + biases['bd1']))
	h_fc1_drop = tf.nn.dropout(h_fc1, dropout)

	# Output, landmark regression
	y_landmark = tf.matmul(h_fc1_drop, weights['out']) + biases['out']
	W_fc_landmark = weights['out']
	return y_landmark, W_fc_landmark

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 16 outputs
    'wc1': weight_variable([5, 5, 1, 16]),
    # 3x3 conv, 16 inputs, 48 outputs
    'wc2': weight_variable([3, 3, 16, 48]),
	# 3x3 conv, 48 inputs, 64 outputs
    'wc3': weight_variable([3, 3, 48, 64]),
    # 2x2 conv, 64 inputs, 64 outputs
    'wc4': weight_variable([2, 2, 64, 64]),

    # fully connected, 2*2*64 inputs, 100 outputs
    'wd1': weight_variable([2 * 2 * 64, 100]),
    # 100 inputs, 68*2 outputs (class prediction)
    'out': weight_variable([100, landmark_num])
}

biases = {
    'bc1': bias_variable([16]),
    'bc2': bias_variable([48]),
    'bc3': bias_variable([64]),
    'bc4': bias_variable([64]),

    'bd1': bias_variable([100]),
    'out': bias_variable([landmark_num])
}

# Construct model
y_landmark, W_fc_landmark = conv_net(image, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_sum(tf.square(landmark - y_landmark)) / 2 + 2*tf.nn.l2_loss(W_fc_landmark)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, lr_iter, lr_base, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# Evaluate model
#landmark_error = 1 / 2 * tf.reduce_sum(tf.square(landmark - y_landmark))
landmark_error = tf.nn.l2_loss(landmark - y_landmark) / 2
fc_landmark_error = 2*tf.nn.l2_loss(W_fc_landmark)

saver = tf.train.Saver()  # defaults to saving all variables

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
add_global = global_step.assign_add(1)

with tf.Session() as sess:
	sess.run(init)
	step = 0
	#saver.restore(sess, checkpoint_dir + "model.ckpt-201")
	# Keep training until reach max iterations
	if isTrain:
		while step < training_iters:
			step = sess.run(global_step)
			img, land = get_batch(batch_size)

			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={image: img, landmark: land, keep_prob: dropout})
			if step % display_step == 0:
				# Calculate batch loss and accuracy
				loss, landerror, fc_err, lr = sess.run([cost, landmark_error, fc_landmark_error, learning_rate], feed_dict={image: img, landmark: land, keep_prob: 1.})
				print("Iter " + str(step) + ", Minibatch Loss= " + \
	                  "{:.6f}".format(loss) + ", Landmark error= " + \
	                  "{:.5f}".format(landerror) + ", fc error= " + \
	                  "{:.5f}".format(fc_err) + ", lr= " + "{:.6f}".format(lr))
			if step % checkpoint_steps == 0:
				#save network
				saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=step)
		print("Optimization Finished!")

	# Calculate accuracy for 200 test images
	testimg, testland, testindex = get_testset(testbatch)
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir) 
	fileopen = open("landmarks.txt", 'w')
	if ckpt and ckpt.model_checkpoint_path:  
		saver.restore(sess, ckpt.model_checkpoint_path) 
	else:  
		pass
	out_landmark = sess.run(y_landmark, feed_dict={image: testimg, keep_prob: 1.})
	#fileopen.write(out_landmark)
	print("Testing landmark:", out_landmark)
	np.savetxt('out_landmark.txt',out_landmark)
	np.savetxt('out_index.txt',testindex)
	#fileopen.close()
    	#print("Testing landmark:", sess.run(y_landmark, feed_dict={image: testimg, keep_prob: 1.}))

