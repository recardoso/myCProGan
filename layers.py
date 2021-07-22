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
        self.shape = input_shape_value #[kernel, kernel, x.shape[1]]
        self.fan_in = tf.cast(np.prod(self.shape), tf.float32)
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
        self.bias = tf.Variable(initial_value = bias_init(shape=(input_shape[1],), dtype='float32'), trainable=True)  

    def call(self, inputs, **kwargs):
        if len(inputs.shape) == 2:
            return inputs + self.bias
        else:
            return inputs + tf.reshape(self.bias, [1, -1, 1, 1])
    

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

def conv2dwscale(x, filters, kernel_size, gain, use_pixelnorm=False, activation=False, strides=(1,1), name=''):
    init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.) #change the stddev to 1
    #[kernel, kernel, x.shape[1]]
    #x = WeightScaleLayer(input_shape_value=(kernel_size[0], kernel_size[1], x.shape[1]), gain=gain)(x)
    x = Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32', data_format='channels_first', name=name)(x)
    x = WeightScaleLayer(input_shape_value=(kernel_size[0], kernel_size[1], x.shape[1]), gain=gain)(x)
    bias_name = name + '_bias' 
    x = Bias(input_shape=x.shape, name=bias_name)(x)
    if activation:
        x = LeakyReLU(0.2)(x) #.2 used in the oroginal function
    if use_pixelnorm:
        x = Pixel_norm_layer(epsilon=1e-8)(x)
    return x 

def densewscale(x, filters, gain, use_pixelnorm=False, activation=False, name='', extra_bias=False):
    init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.) #change the stddev to 1
    if len(x.shape) > 2: #is this necessary or the dense layer already does this?
        x = tf.reshape(x, [-1, np.prod(x.shape[1:])])
    #[x.shape[1].value, fmaps]
    #x = WeightScaleLayer(input_shape_value=(x.shape[1]), gain=gain)(x)
    x = Dense(filters, use_bias=False, kernel_initializer=init, dtype='float32', name=name)(x)
    x = WeightScaleLayer(input_shape_value=(x.shape[1]), gain=gain)(x)
    bias_name = name + '_bias'
    bias = Bias(input_shape=x.shape, name=bias_name)
    x = bias(x)
    if activation:
        x = LeakyReLU(0.2)(x) #.2 used in the oroginal function
    if use_pixelnorm:
        #x = PN(act(apply_bias(x)))
        if extra_bias:
            x = bias(x)
            x = LeakyReLU(0.2)(x)
        x = Pixel_norm_layer(epsilon=1e-8)(x)
    return x 

def conv2d_downscale2dwscale(x, filters, kernel_size, gain, use_pixelnorm=False, activation=False, strides=(1,1), name=''):
    init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.) #change the stddev to 1
    #[kernel, kernel, x.shape[1]]
    #x = WeightScaleLayer(input_shape_value=(kernel_size[0], kernel_size[1], x.shape[1]), gain=gain)(x)
    x = Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32', data_format='channels_first', name=name)(x)
    x = WeightScaleLayer(input_shape_value=(kernel_size[0], kernel_size[1], x.shape[1]), gain=gain)(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(x) #is this in the correct position?
    bias_name = name + '_bias' 
    x = Bias(input_shape=x.shape, name=bias_name)(x)
    if activation:
        x = LeakyReLU(0.2)(x) #.2 used in the oroginal function
    if use_pixelnorm:
        x = Pixel_norm_layer(epsilon=1e-8)(x)
    return x 

def upscale_conv2dwscale(x, filters, kernel_size, gain, use_pixelnorm=False, activation=False, strides=(1,1), name=''):
    init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.) #change the stddev to 1
    #[kernel, kernel, x.shape[1]]
    #x = WeightScaleLayer(input_shape_value=(kernel_size[0], kernel_size[1], x.shape[1]), gain=gain)(x)
    x = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(x) #is this in the correct position?
    x = Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32', data_format='channels_first', name=name)(x)
    x = WeightScaleLayer(input_shape_value=(kernel_size[0], kernel_size[1], x.shape[1]), gain=gain)(x)
    bias_name = name + '_bias' 
    x = Bias(input_shape=x.shape, name=bias_name)(x)
    if activation:
        x = LeakyReLU(0.2)(x) #.2 used in the oroginal function
    if use_pixelnorm:
        x = Pixel_norm_layer(epsilon=1e-8)(x)
    return x 


def fromrgb(x, res, filters=3, gain=np.sqrt(2)): #filters is the number of channels
    name = '%dx%d/FromRGB_lod' % (2**res, 2**res)
    return conv2dwscale(x, filters, kernel_size=(1,1), gain=gain, use_pixelnorm=False, activation=True, strides=(1,1), name=name)

def torgb(x, res, filters=3, gain=1):
    # lod = resolution_log2 - res
    # with tf.variable_scope('ToRGB_lod%d' % lod):
    name = '%dx%d/ToRGB_lod' % (2**res, 2**res)
    return conv2dwscale(x, filters, kernel_size=(1,1), gain=gain, use_pixelnorm=False, activation=False, strides=(1,1), name=name)

def encoder(x, resolution_log2, gain=np.sqrt(2)):
    x = fromrgb(x, resolution_log2)
    y = x
    for res in range(resolution_log2, 4, -1):
        name = '%dx%d/Conv_down%d' % (2**resolution_log2, 2**resolution_log2, res)
        x = conv2d_downscale2dwscale(x, filters=128, kernel_size=(3,3), gain=gain, use_pixelnorm=False, activation=True, strides=(1,1), name=name)
    name ='Conv_down4'
    x = conv2d_downscale2dwscale(x, filters=32, kernel_size=(3,3), gain=gain, use_pixelnorm=False, activation=True, strides=(1,1), name=name)
    name = 'Dense0up'
    x = densewscale(x, 2048, gain=gain, use_pixelnorm=True, activation=True, name=name, extra_bias=True)
    name = 'Dense1up'
    x = densewscale(x, 2048, gain=1, use_pixelnorm=False, activation=True, name=name)
    return x, y

def nf(stage, fmap_base, fmap_decay, fmap_max): 
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
     
def generator_block(x,res,resolution_log2,gain=np.sqrt(2),y=None):
    fmap_base           = 8192        # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 128          # Maximum number of feature maps in any layer.
    scope_name = '%dx%d' % (2**res, 2**res)
    # y_temp=y
    if res == 5: # 32x32
        x = Pixel_norm_layer(epsilon=1e-8)(x)
        x = tf.reshape(x, [-1, 32, 8, 8]) #if res is changed this should change as well
        bias_name = scope_name + '_Dense_bias_PN'
        #x = PN(act(apply_bias(x)))
        x = Bias(input_shape=x.shape, name=bias_name)(x)
        x = LeakyReLU(0.2)(x)
        x = Pixel_norm_layer(epsilon=1e-8)(x)
        name = scope_name + '/Conv'
        x = upscale_conv2dwscale(x, filters=64, kernel_size=(3,3), gain=gain, use_pixelnorm=True, activation=True, strides=(1,1), name=name) #original uses a conv2d for 4x4 layer
    else: #x64 and up
        # for r in range(resolution_log2, res-1, -1):
        #     name = scope_name + 'Conv_down%d_%d' %(res, r)
        #     y_temp= conv2d_downscale2dwscale(y_temp, filters=nf(res-5, fmap_base, fmap_decay, fmap_max), kernel_size=(3,3), gain=gain, use_pixelnorm=False, activation=True, strides=(1,1), name=name)
        x = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(x)
        # x = tf.concat([x, y_temp], axis=1)
        # name = scope_name + '/ConvConcat'
        # x = conv2dwscale(x, filters=nf(res-5, fmap_base, fmap_decay, fmap_max), kernel_size=(3,3), gain=gain, use_pixelnorm=True, activation=True, strides=(1,1), name=name)
        name = scope_name + '/Conv0'
        x = conv2dwscale(x, filters=nf(res-5, fmap_base, fmap_decay, fmap_max), kernel_size=(3,3), gain=gain, use_pixelnorm=True, activation=True, strides=(1,1), name=name)
        name = scope_name + '/Conv1'
        x = conv2dwscale(x, filters=nf(res-5, fmap_base, fmap_decay, fmap_max), kernel_size=(3,3), gain=gain, use_pixelnorm=True, activation=True, strides=(1,1), name=name)
    return x

def discriminator_block(x,res,resolution_log2,gain=np.sqrt(2),mbstd_group_size = 4):
    fmap_base           = 8192         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512          # Maximum number of feature maps in any layer.
    label_size          = 0
    scope_name = '%dx%d' % (2**res, 2**res)
    if res >= 3: # 8x8 and up
        name = scope_name + '/Conv0'
        x = conv2dwscale(x, nf(res-1, fmap_base, fmap_decay, fmap_max), kernel_size=(3,3), gain=gain, use_pixelnorm=False, activation=True, strides=(1,1), name=name)
        #fused?
        name = scope_name + '/Conv1_down'
        x = conv2d_downscale2dwscale(x, filters=nf(res-2, fmap_base, fmap_decay, fmap_max), kernel_size=(3,3), gain=gain, use_pixelnorm=False, activation=True, strides=(1,1), name=name)
    else: # 4x4
        if mbstd_group_size > 1:
            x = Minibatch_stddev_layer(mbstd_group_size)(x)
        name = scope_name + '/Conv0'
        x = conv2dwscale(x, filters=nf(res-1, fmap_base, fmap_decay, fmap_max), kernel_size=(3,3), gain=gain, use_pixelnorm=False, activation=True, strides=(1,1), name=name)
        name = scope_name + '/Dense0'
        x = densewscale(x, filters=nf(res-2, fmap_base, fmap_decay, fmap_max), gain=gain, use_pixelnorm=False, activation=True, name=name)

        name = scope_name + '/Dense1'
        x = densewscale(x, filters=1+label_size, gain=1, use_pixelnorm=False, activation=True, name=name)     
    return x

def transform_generator_input(x,res,resolution_log2,gain=np.sqrt(2),norm=True):
    fmap_base           = 8192        # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 128          # Maximum number of feature maps in any layer.
    scope_name = '%dx%d' % (2**res, 2**res)

    if norm:
        x = Pixel_norm_layer(epsilon=1e-8)(x)
        name = scope_name + '/Dense_Input'
    # original Cprogan used 2048 (128*16), the original uses 8192 (512*16) 
    # needs to change the 32 on the first reshape in case this is changed
    x = densewscale(x, filters=128*16, gain=gain/4, use_pixelnorm=False, activation=False, name=name)

    return x



        
