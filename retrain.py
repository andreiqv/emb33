#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
in 

"""

# export CUDA_VISIBLE_DEVICES=1

from __future__ import absolute_import,  division, print_function
import tensorflow as tf
import sys
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)

#import load_data
import _pickle as pickle
import gzip

#import tensorflow_hub as hub

from rotate_images import *

BATCH_SIZE = 16
NUM_ITERS = 500000

data_file = "dump.gz"
f = gzip.open(data_file, 'rb')
data = pickle.load(f)
#data_1 = load_data(in_dir, img_size=(540,540))
#data = split_data(data1, ratio=(6,1,3))

train = data['train']
valid = data['valid']
test  = data['test']
train_data = train['embedding']
valid_data = valid['embedding']
test_data = test['embedding']
train_labels = train['labels']
valid_labels = valid['labels']
test_labels = test['labels']

print('train size:', train['size'])
print('valid size:', valid['size'])
print('test size:', test['size'])
im0 = train_data[0]
print('Data was loaded.')
print(im0.shape)
#sys.exit()

#train_data = [np.transpose(t) for t in train_data]
#valid_data = [np.transpose(t) for t in valid_data]
#test_images = [np.transpose(t) for t in test_images]
num_train_batches = train['size'] // BATCH_SIZE
num_valid_batches = valid['size'] // BATCH_SIZE
num_test_batches = test['size'] // BATCH_SIZE
print('num_train_batches:', num_train_batches)
print('num_valid_batches:', num_valid_batches)
print('num_test_batches:', num_test_batches)

SAMPLE_SIZE = train['size']
min_valid_accuracy = 1000


# some functions

def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W, name=None):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',  name=name) 

def max_pool_3x3(x, name=None):
	return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME',  name=name)


def convPoolLayer(p_in, kernel, pool_size, num_in, num_out, func=None, name=''):
	W = weight_variable([kernel[0], kernel[1], num_in, num_out], name='W'+name)  # 32 features, 5x5
	b = bias_variable([num_out], name='b'+name)
	
	if func:
		h = func(conv2d(p_in, W, name='conv'+name) + b, name='relu'+name)
	else:
		h = conv2d(p_in, W, name='conv'+name) + b

	if pool_size == 2:
		p_out = max_pool_2x2(h, name='pool'+name)
	elif pool_size == 3:
		p_out = max_pool_3x3(h, name='pool'+name)
	else:
		raise("bad pool size")
	print('p{0} = {1}'.format(name, p_out))
	return p_out

def fullyConnectedLayer(p_in, input_size, num_neurons, func=None, name=''):
	num_neurons_6 = 128
	W = weight_variable([input_size, num_neurons], name='W'+name)
	b = bias_variable([num_neurons], name='b'+name)
	if func:
		h = func(tf.matmul(p_in, W) + b, name='relu'+name)
	else:
		h = tf.matmul(p_in, W) + b
	print('h{0} = {1}'.format(name, h))
	return h


#------------------------

def network1(input_tensor, input_size):

	f1 = fullyConnectedLayer(
		input_tensor, input_size=bottleneck_tensor_size, num_neurons=1, 
		func=tf.nn.sigmoid, name='F1') # func=tf.nn.relu
	
	return f1


def network2(input_tensor, input_size, hidden_num=512):

	f1 = fullyConnectedLayer(
		input_bottleneck, input_size=bottleneck_tensor_size, num_neurons=hidden_num, 
		func=tf.nn.relu, name='F1') # func=tf.nn.relu
	
	drop1 = tf.layers.dropout(inputs=f1, rate=0.4)	
	
	f2 = fullyConnectedLayer(drop1, input_size=hidden_num, num_neurons=1, 
		func=tf.nn.sigmoid, name='F2')

	return f2


#-------------------

# Create a new graph
graph = tf.Graph() # no necessiry

with graph.as_default():

	# 1. Construct a graph representing the model.
	bottleneck_tensor_size =  2048
	x = tf.placeholder(tf.float32, [None, 1, bottleneck_tensor_size], name='Placeholder-x') # Placeholder for input.
	y = tf.placeholder(tf.float32, [None, 1], name='Placeholder-y')   # Placeholder for labels.
	
	input_bottleneck = tf.reshape(x, [-1, bottleneck_tensor_size])

	output = network2(input_bottleneck, bottleneck_tensor_size)
	print('output =', output)

	# 2. Add nodes that represent the optimization algorithm.

	loss = tf.reduce_mean(tf.square(output - y))	
	#loss = tf.reduce_sum(tf.pow(output - y, 2))/(n_instances)
	#loss = tf.reduce_mean(tf.squared_difference(output, y))
	#loss = tf.nn.l2_loss(output - y)

	#optimizer = tf.train.AdagradOptimizer(0.01)
	optimizer= tf.train.AdagradOptimizer(0.005)
	#optimizer= tf.train.AdamOptimizer(0.01)
	#train_op = tf.train.GradientDescentOptimizer(0.01)
	train_op = optimizer.minimize(loss)

	# 3. Execute the graph on batches of input data.
	with tf.Session() as sess:  # Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)	# Randomly initialize weights.

		for iteration in range(NUM_ITERS):			  # Train iteratively for NUM_iterationS.		 

			if iteration % 5000 == 0:

				output_values = output.eval(feed_dict = {x:valid_data[:3]})
				print('valid: {0:.2f} - {1:.2f}'.format(output_values[0][0]*360, valid_labels[0][0]*360))
				print('valid: {0:.2f} - {1:.2f}'.format(output_values[1][0]*360, valid_labels[1][0]*360))

				print('filenames:', valid['filenames'][0])
				print('labels:', valid['labels'][0][0], '   grad=', valid['labels'][0][0]*360.0)
				print('output:', output_values[0][0], '   grad=', output_values[0][0]*360.0)
				print('emb:', valid['embedding'][0])

				output_angles_valid = []
				for i in range(num_valid_batches):
					feed_dict = {x:valid_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]}
					#print(feed_dict)
					output_values = output.eval(feed_dict=feed_dict)
					#print(i, output_values)
					#print(output_values.shape)
					t = [output_values[i][0]*360.0 for i in range(output_values.shape[0])]
					#print(t)
					output_angles_valid += t
				print(output_angles_valid[:min(len(valid_data),10)])


			if iteration % 200 == 0:

				train_accuracy = np.mean( [loss.eval( \
					feed_dict={x:train_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
					y:train_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
					for i in range(0, num_train_batches)])
				
				valid_accuracy = np.mean([ loss.eval( \
					feed_dict={x:valid_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
					y:valid_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
					for i in range(0, num_valid_batches)])

				if valid_accuracy < min_valid_accuracy:
					min_valid_accuracy = valid_accuracy

				min_in_grad = math.sqrt(min_valid_accuracy) * 360.0
				print('iter {0:3}: train_loss={1:0.4f}, valid_loss={2:0.4f} (min={3:0.4f} ({4:0.2f} gr.))'.\
					format(iteration, train_accuracy, valid_accuracy, min_valid_accuracy, min_in_grad))

				"""
				#train_accuracy = loss.eval(feed_dict = {x:train_data[0:BATCH_SIZE], y:train_labels[0:BATCH_SIZE]})
				#valid_accuracy = loss.eval(feed_dict = {x:valid_data[0:BATCH_SIZE], y:valid_labels[0:BATCH_SIZE]})
				"""
			
			# TRAIN:
			a1 = iteration*BATCH_SIZE % train['size']
			a2 = (iteration + 1)*BATCH_SIZE % train['size']
			x_data = train_data[a1:a2]
			y_data = train_labels[a1:a2]

			if len(x_data) <= 0: continue
			sess.run(train_op, 
				feed_dict={x: x_data, y: y_data})  # Perform one training iteration.		

		# Save the comp. graph

		"""
		x_data, y_data =  valid_data, valid_labels #mnist.train.next_batch(BATCH_SIZE)		
		writer = tf.summary.FileWriter("output", sess.graph)
		print(sess.run(train_op, feed_dict={x: x_data, y: y_data}))
		writer.close()  
		"""

		# Test of model
		"""
		HERE SOME ERROR ON GPU OCCURS
		num_test_batches = test['size'] // BATCH_SIZE
		test_accuracy = np.mean([ loss.eval( \
			feed_dict={x:test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE]}) \
			for i in range(num_test_batches) ])
		print('Test of model')
		print('Test_accuracy={0:0.4f}'.format(test_accuracy))
		"""

		"""
		test_accuracy = loss.eval(feed_dict={x:test_images[0:BATCH_SIZE]})
		print('Test_accuracy={0:0.4f}'.format(test_accuracy))				
		"""

		# Rotate images:
		in_dir = 'data'
		out_dir = 'valid'
		file_names = valid['filenames']
		angles = output_angles_valid
		rotate_images_with_angles(in_dir, out_dir, file_names, angles)
		
		"""
		# Saver
		saver = tf.train.Saver()		
		saver.save(sess, './save_model/my_test_model')  
		"""


