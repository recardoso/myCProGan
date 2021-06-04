import numpy as np
import tensorflow as tf

# NOTE: Do not import any application-specific modules here!

#----------------------------------------------------------------------------

import os
import sys
import inspect
import importlib
import imp
from collections import OrderedDict
from tensorflow.python.ops import nccl_ops

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Layer, InputSpec, Conv2D, Conv2DTranspose, Activation, Reshape, LayerNormalization, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import Input, UpSampling2D, Dropout, Concatenate, Add, Dense, Multiply, LeakyReLU, Flatten, AveragePooling2D, Multiply

from layers import *

#Use for tensorflow 2
from tensorflow.python.keras.layers.ops import core as core_ops

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

def generator(resolution=256,num_channels=3,num_replicas=1): #confirm params and confirm initializers strides bias

    resolution_log2 = int(np.log2(resolution))
    dshape=( 3, resolution, resolution) 
    latent = Input(shape=dshape)
    latent = tf.cast(latent, tf.float32)
    lod_in = Input(shape=(), batch_size=num_replicas, name='lod_in')
    lod_in = tf.cast(lod_in, tf.float32)
    dtype = 'float32'

    x,y = encoder(latent, resolution_log2, gain=np.sqrt(2))



    x = generator_block(x, 5, y, resolution_log2)
    images_out = torgb(x, 5)
    for res in range(6, resolution_log2 + 1):
        lod = resolution_log2 - res
        x = generator_block(x, res, y, resolution_log2)
        img = torgb(x, res)
        images_out = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(images_out)
        images_out = Lerp_clip_layer()(img, images_out, lod_in - lod)
        #lerp_clip(img, images_out, lod_in - lod)

    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    
    Model(inputs=[latent, lod_in], outputs=[images_out]).summary()
    
    return Model(inputs=[latent, lod_in], outputs=[images_out])

def discriminator(resolution=256, num_channels=3, label_size = 0, mbstd_group_size = 4, old_res=128, num_replicas=1):
    resolution_log2 = int(np.log2(resolution))
    dshape=( num_channels, resolution, resolution) 
    normal_gain = np.sqrt(2)
    images_in = Input(shape=dshape)
    images_in = tf.cast(images_in, tf.float32)
    lod_in= Input(shape=(), batch_size=num_replicas, name='lod_in')
    lod_in = tf.cast(lod_in, tf.float32)

    #TODO: Final 2 dense layers too sharp: add one or more dense layers in the middle
    dtype               = 'float32'

    img = images_in
    x = fromrgb(img, resolution_log2)
    for res in range(resolution_log2, 2, -1):
        lod = resolution_log2 - res
        x = discriminator_block(x, res, resolution_log2, mbstd_group_size = mbstd_group_size)
        img = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(img)
        y = fromrgb(img, res - 1)
        x = Lerp_clip_layer()(x, y, lod_in - lod)
    combo_out = discriminator_block(x, 2, resolution_log2, mbstd_group_size = mbstd_group_size)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    #labels_out = tf.identity(combo_out[:, 1:], name='labels_out')

    Model(inputs=[images_in, lod_in], outputs=[scores_out]).summary()
    
    return Model(inputs=[images_in, lod_in], outputs=[scores_out])