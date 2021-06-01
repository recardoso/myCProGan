#defenition of possible layers

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

#Use for tensorflow 2
from tensorflow.python.keras.layers.ops import core as core_ops

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------

class WeightScaleLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape_value, gain = np.sqrt(2), **kwargs):
        super(WeightScaleLayer, self).__init__(**kwargs)
        self.shape = input_shape_value
        self.fan_in = np.prod(self.shape)
        #self.wscale = gain / np.sqrt(self.fan_in)
        self.wscale = gain * tf.math.rsqrt(self.fan_in)
      
    def call(self, inputs, **kwargs):
        #inputs = tf.cast(inputs, tf.float32)
        return inputs * self.wscale
    

class Bias(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        bias_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value = bias_init(shape=(input_shape[-1],), dtype='float32'), trainable=True)  

    def call(self, inputs, **kwargs):
        return inputs + self.bias
    

class Lerp_clip_layer(tf.keras.layers.Add):
    def __init__(self, **kwargs):
        super(Lerp_clip_layer, self).__init__(**kwargs)

    def _merge_function(self, inputs):
        assert (len(inputs) == 3)
        a = inputs[0]
        b = inputs[1]
        t = tf.clip_by_value(inputs[2], 0.0, 1.0)
        output = a + (b - a) * t
        return output
        


class Minibatch_stddev_layer(tf.keras.layers.Layer):
    def __init__(self, group_size=4):
        super(Minibatch_stddev_layer, self).__init__()
        self.group_size = group_size

    def call(self, inputs):
        #with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(self.group_size, tf.shape(inputs)[0])       # Minibatch must be divisible by (or smaller than) group_size.
        s = inputs.shape                                                    # [NCHW]  Input shape.
        y = tf.reshape(inputs, [group_size, -1, s[1], s[2], s[3]])          # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                                          # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)                       # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                            # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                               # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)                  # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, inputs.dtype)                                        # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])                         # [N1HW]  Replicate over group and pixels.
        return tf.concat([inputs, y], axis=1)       
        

class Pixel_norm_layer(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8):
        super(Pixel_norm_layer, self).__init__()
        self.epsilon = epsilon

    def call(self, inputs): #rsqrt or sqrt
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + self.epsilon)  

def conv2dwscale(x, filters, kernel_size, gain, use_pixelnorm=False, activation=False, strides=(1,1)):
    x = WeightScaleLayer(shape=(kernel_size[0], kernel_size[1], in_filters), gain=gain)(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32')(x)
    x = Bias(input_shape=x.shape)(x)
    if activation:
        x = layers.LeakyReLU(0.2)(x)
    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x 

def WeightScalingConv(x, filters, kernel_size, gain, use_pixelnorm=False, activate=None, strides=(1,1)):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = backend.int_shape(x)[-1]
    x = layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(kernel_size[0], kernel_size[1], in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate=='LeakyReLU':
        x = layers.LeakyReLU(0.2)(x)
    elif activate=='tanh':
        x = layers.Activation('tanh')(x)
    
    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x 


def fromrgb(a):
    return a