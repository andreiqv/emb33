#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import tensorflow as tf
import tensorflow_hub as hub



"""
def module1(x, shape):

	fullconn_input_size = shape[0] * shape[1] * shape[2]
	p_flat = tf.reshape(x, [-1, fullconn_input_size])
	f1 = fullyConnectedLayer(p_flat, input_size=fullconn_input_size, num_neurons=1024, 
		func=tf.nn.relu, name='M_F1')
	
	drop1 = tf.layers.dropout(inputs=f1, rate=0.4)	
	f2 = fullyConnectedLayer(drop1, input_size=1024, num_neurons=1024, 
		func=tf.nn.relu, name='M_F2')
	
	drop2 = tf.layers.dropout(inputs=f2, rate=0.4)	
	f3 = fullyConnectedLayer(drop2, input_size=1024, num_neurons=1001, 
		func=tf.sigmoid, name='M_F3')

	return f3
"""


def calculate_embedding(images, shape):

	#bottleneck_tensor_size = 1024

	height, width, color =  shape

	x = tf.placeholder(tf.float32, [None, height, width, 3], name='Placeholder-x')

	resized_input_tensor = tf.reshape(x, [-1, height, width, 3])

	#module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1")	
	
	module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1")	 
		# num_features = 2048, height x width = 224 x 224 pixels
	assert height, width == hub.get_expected_image_size(module)
	
	bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]
	print('bottleneck_tensor:', bottleneck_tensor)

	with tf.Session() as sess:  # Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)	# Randomly initialize weights.
		
		embedding = [bottleneck_tensor.eval(\
					feed_dict={ x : images[i:i+1] })\
					for i in range(len(images))]	

	print(len(embedding))				
	return embedding
	