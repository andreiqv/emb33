#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
in 

"""

# export CUDA_VISIBLE_DEVICES=1

from __future__ import absolute_import,  division, print_function
import os
import sys
import argparse
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)

#import load_data
import _pickle as pickle
import gzip

import tensorflow as tf
#import tensorflow_hub as hub

from rotate_images import *
from layers import *

HIDDEN_NUM = 8

if os.path.exists('.notebook'):
	bottleneck_tensor_size =  588
	BATCH_SIZE = 3
	DISPLAY_INTERVAL, NUM_ITERS = 1, 50
else:
	bottleneck_tensor_size =  2048
	BATCH_SIZE = 10
	DISPLAY_INTERVAL, NUM_ITERS = 100, 20*1000*1000

to_deg = lambda x : math.sqrt(x) * 360.0

"""
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
"""

#------------------------

def network1(input_tensor, input_size):

	f1 = fullyConnectedLayer(
		input_tensor, input_size=bottleneck_tensor_size, num_neurons=1, 
		func=tf.nn.sigmoid, name='F1') # func=tf.nn.relu
	
	return f1


def network2(input_tensor, input_size, hidden_num=HIDDEN_NUM):

	f1 = fullyConnectedLayer(
		input_bottleneck, input_size=bottleneck_tensor_size, num_neurons=hidden_num, 
		func=tf.nn.relu, name='F1') # func=tf.nn.relu
	
	drop1 = tf.layers.dropout(inputs=f1, rate=0.4)	
	
	f2 = fullyConnectedLayer(drop1, input_size=hidden_num, num_neurons=1, 
		func=tf.nn.sigmoid, name='F2')

	return f2



def createParser ():
	"""
	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--in_file', default="dump.gz", type=str,\
		help='input dir')
	parser.add_argument('-k', '--k', default=2, type=int,\
		help='number of network')
	parser.add_argument('-hn', '--hidden_num', default=8, type=int,\
		help='number of neurons in hiden layer')

	return parser




if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	data_file = arguments.in_file

	if arguments.k == 1:	
		neural_network = network1
	elif arguments.k == 2:	
		neural_network = network2
	else:
		raise Exception('Bad argument arguments.k')

	if arguments.hidden_num > 0:
		HIDDEN_NUM = arguments.hidden_num


	#data_1 = load_data(in_dir, img_size=(540,540))
	#data = split_data(data1, ratio=(6,1,3))

	print('data_file =', data_file)
	print('selected network =', arguments.k)
	if arguments.k > 1:
		print('HIDDEN_NUM =', HIDDEN_NUM)

	f = gzip.open(data_file, 'rb')
	data = pickle.load(f)
	train = data['train']
	valid = data['valid']
	test  = data['test']
	#train_data = train['embedding']
	#valid_data = valid['embedding']
	#test_data = test['embedding']
	train_data = train['images']
	valid_data = valid['images']
	test_data = test['images']
	train_labels = train['labels']
	valid_labels = valid['labels']
	test_labels = test['labels']
	train['size'] = len(train['labels'])
	valid['size'] = len(valid['labels'])
	test['size'] = len(test['labels'])

	print('train size:', len(train['labels']))
	print('valid size:', len(valid['labels']))
	print('test size:', len(valid['labels']))
	print('Data was loaded.')
	print('Example of data:', train_data[0].shape)
	print('Example of label:',train_labels[0])
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
	min_valid_loss = 1000


	#-------------------

	# Create a new graph
	graph = tf.Graph() # no necessiry

	with graph.as_default():

		# 1. Construct a graph representing the model.
		
		x = tf.placeholder(tf.float32, [None, 1, bottleneck_tensor_size], name='Placeholder-x') # Placeholder for input.
		y = tf.placeholder(tf.float32, [None, 1], name='Placeholder-y')   # Placeholder for labels.
		
		input_bottleneck = tf.reshape(x, [-1, bottleneck_tensor_size])

		output = neural_network(input_bottleneck, bottleneck_tensor_size)
		print('output =', output)

		# 2. Add nodes that represent the optimization algorithm.
		loss = tf.reduce_mean(tf.square(output - y))
		#loss = tf.reduce_mean(tf.abs(1 -  tf.abs(tf.abs(output - y) - 1 ))) # 
		#loss = tf.reduce_mean(tf.squared_difference(y, output))
		#loss = tf.nn.l2_loss(output - y)
		#loss = tf.losses.mean_squared_error(labels=y, predictions=output)
		
		#optimizer = tf.train.AdagradOptimizer(0.01)
		optimizer= tf.train.AdagradOptimizer(0.005)
		#optimizer= tf.train.AdamOptimizer(0.005)
		#train_op = tf.train.GradientDescentOptimizer(0.01)
		train_op = optimizer.minimize(loss)
			
		# for classification:
		#loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
		#train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
		#correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		output_angles_valid = []

		# 3. Execute the graph on batches of input data.
		with tf.Session() as sess:  # Connect to the TF runtime.
			init = tf.global_variables_initializer()
			sess.run(init)	# Randomly initialize weights.
			for iteration in range(NUM_ITERS):			  # Train iteratively for NUM_iterationS.		 

				if iteration % (100*DISPLAY_INTERVAL) == 0:

					#output_values = output.eval(feed_dict = {x:train['images'][:3]})
					#print('train: {0:.2f} - {1:.2f}'.format(output_values[0][0]*360, train['labels'][0][0]*360))
					#print('train: {0:.2f} - {1:.2f}'.format(output_values[1][0]*360, train['labels'][1][0]*360))
					output_values = output.eval(feed_dict = {x:valid['images'][:3]})
					print('valid: {0:.2f} - {1:.2f}'.format(output_values[0][0]*360, valid['labels'][0][0]*360))
					print('valid: {0:.2f} - {1:.2f}'.format(output_values[1][0]*360, valid['labels'][1][0]*360))
					#print('valid: {0:.2f} - {1:.2f}'.format(output_values[2][0]*360, valid['labels'][2][0]*360))
					
					output_angles_valid = []
					for i in range(num_valid_batches):
						feed_dict = {x:valid['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}
						#print(feed_dict)
						output_values = output.eval(feed_dict=feed_dict)
						#print(i, output_values)
						#print(output_values.shape)
						t = [output_values[i][0]*360.0 for i in range(output_values.shape[0])]
						#print(t)
						output_angles_valid += t
					print(output_angles_valid[:max(len(valid),10)])


				if iteration % (10*DISPLAY_INTERVAL) == 0:

					train_loss = np.mean( [loss.eval( \
						feed_dict={x:train['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
						y:train['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
						for i in range(0,num_train_batches)])
					valid_loss = np.mean([ loss.eval( \
						feed_dict={x:valid['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
						y:valid['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
						for i in range(0,num_valid_batches)])

					if valid_loss < min_valid_loss:
						min_valid_loss = valid_loss

					#min_in_grad = math.sqrt(min_valid_loss) * 360.0
					epoch = iteration//(num_train_batches // BATCH_SIZE * BATCH_SIZE)
					print('epoch {0:2} (i={1:06}): train={2:0.2f}, valid={3:0.2f} (min={4:0.2f})'.\
						format(epoch, iteration, to_deg(train_loss), to_deg(valid_loss), to_deg(min_valid_loss)))

					"""
					#train_loss = loss.eval(feed_dict = {x:train['images'][0:BATCH_SIZE], y:train['labels'][0:BATCH_SIZE]})
					#valid_loss = loss.eval(feed_dict = {x:valid['images'][0:BATCH_SIZE], y:valid['labels'][0:BATCH_SIZE]})
					"""
				
				a1 = iteration*BATCH_SIZE % train['size']
				a2 = (iteration + 1)*BATCH_SIZE % train['size']
				x_data = train['images'][a1:a2]
				y_data = train['labels'][a1:a2]
				if len(x_data) <= 0: continue
				sess.run(train_op, {x: x_data, y: y_data})  # Perform one training iteration.		
				#print(a1, a2, y_data)			

			# Save the comp. graph
			if False:
				print('Save the comp. graph')
				x_data, y_data =  valid['images'], valid['labels'] #mnist.train.next_batch(BATCH_SIZE)		
				writer = tf.summary.FileWriter("output", sess.graph)
				print(sess.run(train_op, {x: x_data, y: y_data}))
				writer.close()  

			# Test of model
			"""
			HERE SOME ERROR ON GPU OCCURS
			num_test_batches = test['size'] // BATCH_SIZE
			test_loss = np.mean([ loss.eval( \
				feed_dict={x:test['images'][i*BATCH_SIZE : (i+1)*BATCH_SIZE]}) \
				for i in range(num_test_batches) ])
			print('Test of model')
			print('test_loss={0:0.4f}'.format(test_loss))
			"""

			"""
			print('Test model')
			test_loss = loss.eval(feed_dict={x:test['images'][0:BATCH_SIZE]})
			print('test_loss={0:0.4f}'.format(test_loss))				
			"""

			if False:
				# Rotate images:
				print('Rotate images')
				#in_dir = 'data'
				out_dir = 'valid'
				os.system('mkdir -p {0}'.format(out_dir))
				angles = output_angles_valid
				file_names = valid['filenames'][:len(angles)]
				print('len(angles) =', len(angles))
				print('len(file_names) =', len(file_names))
				rotate_images_with_angles(in_dir, out_dir, file_names, angles)
			
			"""
			# Saver
			saver = tf.train.Saver()		
			saver.save(sess, './save_model/my_test_model')  
			"""
