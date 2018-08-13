#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path
import sys
from PIL import Image, ImageDraw
import _pickle as pickle
import gzip
import random
import numpy as np

import tensorflow as tf
#import tensorflow_hub as hub
import network

np.set_printoptions(precision=4, suppress=True)
#import tensorflow_hub as hub


def load_data(in_dir, img_size):	
	""" each image has form [height, width, 3]
	"""

	data = dict()
	data['filenames'] = []
	data['images'] = []
	data['labels'] = []

	files = os.listdir(in_dir)
	random.shuffle(files)

	for file_name in files:

		file_path = in_dir + '/' + file_name

		#img_gray = Image.open(file_path).convert('L')
		#img = img_gray.resize(img_size, Image.ANTIALIAS)
		img = Image.open(file_path)
		img = img.resize(img_size, Image.ANTIALIAS)
		arr = np.array(img, dtype=np.float32) / 256

		name = ''.join(file_name.split('.')[:-1])
		angle = name.split('_')[-1]
		lable = np.array([float(angle) / 360.0], dtype=np.float64)

		if type(lable[0]) != np.float64:
			print(lable[0])
			print(type(lable[0]))
			print('type(lable)!=float')
			raise Exception('lable type is not float')
			
		print('{0}: [{1:.3f}, {2}]' .format(angle, lable[0], file_name))
		data['images'].append(arr)
		data['labels'].append(lable)
		data['filenames'].append(file_name)


	return data
	#return train, valid, test

"""
def covert_data_to_feature_vector(data):

	module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
	height, width = hub.get_expected_image_size(module)
	assert (height, width) == (299, 299)
	image_feature_vector = module(data['images'])
	data['images'] = image_feature_vector
	return data
"""

#--------------------


def create_bootleneck_data(dir_path, shape):
	""" Calculate feature vectors for rotated images using TF.
	"""
	files = os.listdir(dir_path)
	random.shuffle(files)
	feature_vectors = []
	lables = []

	# Calculate in TF
	height, width, color =  shape
	x = tf.placeholder(tf.float32, [None, height, width, 3], name='Placeholder-x')
	resized_input_tensor = tf.reshape(x, [-1, height, width, 3])
	#module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1")		
	#module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1")
	module = lambda x: network.perceptron(x, shape=shape, output_size=2048)

		# num_features = 2048, height x width = 224 x 224 pixels
	assert height, width == hub.get_expected_image_size(module)	
	bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]
	print('bottleneck_tensor:', bottleneck_tensor)

	with tf.Session() as sess:  # Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)	# Randomly initialize weights.
		
		for file in files:

			print(file)
			file_path = dir_path + '/' + file_name
			img = Image.open(file_path)			

			sx, sy = img.size
			cx, cy = sx/2.0, sy/2.0
			d = min(cx, cy)
			a = d*math.sqrt(2)
			area = (cx - a/2, cy - a/2, cx + a/2, cy + a/2)

			for i in range(0, 12):

				angle = i*30 + randint(0,29)
				print('rotate {0} gradus'.format(angle))

				img_rot = img.rotate(angle)
				box = img_rot.crop(area)
				box = box.resize(img_size, Image.ANTIALIAS)
				arr = np.array(box, dtype=np.float32) / 256
				lable = np.array([float(angle) / 360.0], dtype=np.float64)
				
				feature_vector = bottleneck_tensor.eval(feed_dict={ x : arr })
				feature_vectors.append(feature_vector)
				lables.append(lable)

	print(len(embedding))				
	return embedding


def make_bottleneck_dump(in_dir, out_file, shape):

	bottleneck_data = dict()
	parts = ['train', 'valid', 'test']
	for part in parts:
		part_dir = in_dir + '/' + part		
		bottleneck_data[part] = create_bootleneck_data(part_dir, shape)

	print(len(bottleneck_data['images']))
	print(len(bottleneck_data['labels']))

	
	# save the data on a disk
	"""
	dump = pickle.dumps(bottleneck_data)
	print('dump done')
	f = gzip.open(out_file, 'wb')
	print('gzip done')
	f.write(dump)
	print('dump was written')
	f.close()
	"""

if __name__ == '__main__':

	in_dir = 'data'
	out_file = 'dump.gz'
	shape = 224, 224, 3
	make_bottleneck_dump(in_dir=in_dir, out_file=out_file, shape=shape)
