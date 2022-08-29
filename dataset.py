import os
import time
import numpy as np
import tensorflow as tf
import glob
import argparse
import json
import math

#import config
#import tfutil
#import dataset
import myCProGan.networks2 as networks2
import myCProGan.loss as loss
#from myCProGan.train import *
#import misc
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical, plot_model
#import tensorflow_datasets as tfds

from PIL import Image

import pickle
#import optuna

#----------------------------------------------------------------------------
# Dataset Reading and Processing

def retrieve_balanced_tfrecords(recorddatapaths, batch_size):
    recorddata = tf.data.TFRecordDataset(recorddatapaths, num_parallel_reads=tf.data.experimental.AUTOTUNE)


    retrieveddata = {
        'images': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True), #float32
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        data = tf.io.parse_single_example(example_proto, retrieveddata)
        #data['images'] = tf.reshape(data['images'],[3,256,256])
        #rgb_image = [data['images'],data['images'],data['images']]
        #data['images'] = rgb_image
        #data = tf.io.decode_raw(data['images'], tf.uint8)
        return tf.reshape(data['images'], [3,256,256])
        #return data['images']

    parsed_dataset = recorddata.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(38472).repeat().batch(batch_size, drop_remainder=True)#.with_options(options)
   
    return parsed_dataset#, ds_size

def _parse_function_balanced(example_proto):
        retrieveddata = {
            'images': tf.io.FixedLenSequenceFeature((), dtype=tf.string), #float32
        }
        # Parse the input `tf.Example` proto using the dictionary above.
        data = tf.io.parse_single_example(example_proto, retrieveddata)
        #data['images'] = tf.reshape(data['images'],[3,256,256])
        #rgb_image = [data['images'],data['images'],data['images']]
        #data['images'] = rgb_image
        data = tf.io.decode_raw(data['images'], tf.uint8)
        return tf.reshape(data, [3,256,256])
        #return data['images']

def parse_tfrecord_tf(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
    #data = tf.io.decode_raw(features['data'], tf.uint8)
    data = tf.io.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

def parse_tfrecord_tf_floods(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([2], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
    #data = tf.io.decode_raw(features['data'], tf.uint8)
    data = tf.io.decode_raw(features['data'], tf.uint8)
    return tf.expand_dims(tf.reshape(data, features['shape']), axis=0)

def parse_tfrecord_tf_eurosat(record):
    features = tf.io.parse_single_example(record, features={
        'image': tf.io.FixedLenFeature([], tf.string)})
    #data = tf.io.decode_raw(features['data'], tf.uint8)
    print(features)
    data = tf.io.decode_raw(features['image'], tf.uint8)
    print(data)
    shape = (64, 64, 3)
    data = tf.reshape(data, shape)
    data = tf.transpose(data, [1,2,0])
    return data

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record.numpy())
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    if len(shape)==2:
        return np.expand_dims(np.fromstring(data, np.uint8).reshape(shape), axis=0)
    else:
        return np.fromstring(data, np.uint8).reshape(shape)

def determine_shape(tfr_files, resolution = None):
    tfr_shapes = []
    for tfr_file in tfr_files:
        #tfr_opt = tf.io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE) #be default it shuld be None
        for record in tf.data.TFRecordDataset(tfr_file):
            #print(record)
            tfr_shapes.append(parse_tfrecord_np(record).shape)
            break

            
    # Determine shape and resolution.
    max_shape = max(tfr_shapes, key=lambda shape: np.prod(shape))
    resolution = resolution if resolution is not None else max_shape[1]
    resolution_log2 = int(np.log2(resolution))
    shape = [max_shape[0], resolution, resolution]
    tfr_lods = [resolution_log2 - int(np.log2(shape[1])) for shape in tfr_shapes]
    assert all(shape[0] == max_shape[0] for shape in tfr_shapes)
    assert all(shape[1] == shape[2] for shape in tfr_shapes)
    assert all(shape[1] == resolution // (2**lod) for shape, lod in zip(tfr_shapes, tfr_lods))
    assert all(lod in tfr_lods for lod in range(resolution_log2 - 1))

    return tfr_shapes, tfr_lods


#creates a dataset for each lod (level of detail)
def get_dataset(tfr_files,      # Directory containing a collection of tfrecords files.
    flood= False,
    minibatch_base  = 16,
    minibatch_dict  = {},
    max_minibatch_per_gpu = {},
    num_gpus        = 1,        #number of gpus
    resolution      = None,     # Dataset resolution, None = autodetect.
    label_file      = None,     # Relative path of the labels file, None = autodetect.
    max_label_size  = 0,        # 0 = no labels, 'full' = full labels, <int> = N first label components.
    repeat          = True,     # Repeat dataset indefinitely.
    shuffle_mb      = 4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
    prefetch_mb     = 2048,     # Amount of data to prefetch (megabytes), 0 = disable prefetching.
    buffer_mb       = 256,      # Read buffer size (megabytes).
    num_threads     = 2):       # Number of concurrent threads.

    dtype           = 'uint8'
    #batch_size      = 120       # should be a batch size per each lod??
    
    # I'm not doing anything with the labels I might need to change that

    tfr_shapes, tfr_lods = determine_shape(tfr_files, resolution = resolution)
    _tf_datasets=dict()
    _tf_batch_sizes = dict()

    print(tfr_shapes)
    print(tfr_lods)

    #calculate minibatch size for batching
    # minibatch_base = 16
    # minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
    # max_minibatch_per_gpu = {256: 8, 512: 4, 1024: 2}

    for tfr_file, tfr_shape, tfr_lod in zip(tfr_files, tfr_shapes, tfr_lods):
        if tfr_lod < 0:
            continue

        batch_size = minibatch_dict.get(tfr_shape[-1], minibatch_base)
        batch_size -= batch_size % num_gpus
        if tfr_shape[-1] in max_minibatch_per_gpu:
            batch_size = min(batch_size, max_minibatch_per_gpu[tfr_shape[-1]] * num_gpus)


        #get dataset
        dset = tf.data.TFRecordDataset(tfr_file)#, compression_type='', buffer_size=buffer_mb<<20)
        #use parse function
        if flood:
            dset = dset.map(parse_tfrecord_tf_floods, num_parallel_calls=num_threads)
        else:
            dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)

        #join with labels
        #dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset))
        #create bytes per item
        bytes_per_item = np.prod(tfr_shape) * np.dtype(dtype).itemsize
        #shuffle
        if shuffle_mb > 0:
            dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
        #repeat    
        if repeat:
            dset = dset.repeat()
        #prefetch
        if prefetch_mb > 0:
            dset = dset.prefetch(((prefetch_mb << 20) - 1) // bytes_per_item + 1)
        #batch
        dset = dset.batch(batch_size)
        _tf_datasets[tfr_lod] = dset
        _tf_batch_sizes[tfr_lod] = int(batch_size)

    #iterator = iter(dset)
    
    print(_tf_batch_sizes)

    print('Dataset Read')

    return _tf_datasets, _tf_batch_sizes
