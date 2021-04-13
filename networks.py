# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

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
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        initializer = tf.keras.initializers.RandomNormal() # might need to define the mean and stddev
        values = initializer(shape=shape)
        weight = tf.Variable(shape=shape, initial_value=values) * wscale
        return weight
    else:
        initializer = tf.keras.initializers.RandomNormal(0, std) # might need to define the mean and stddev
        values = initializer(shape=shape)
        weight = tf.Variable(shape=shape, initial_value=values) 
        return weight

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)
    
def dense_tensor(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    w = get_weight([x.shape[1].value, x.shape[2].value, x.shape[3].value], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    initializer = tf.initializers.zeros() # might need to define the mean and stddev
    values = initializer(shape=[x.shape[1]])
    b = tf.Variable(shape=[x.shape[1]], initial_value=values)
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    #with tf.variable_scope('PixelNorm'):
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4):
    #with tf.variable_scope('MinibatchStddev'):
    group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
    y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
    y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
    y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------

#New dense class for the learning rate equalizer
#inherits from dense layer

class DenseLReq(Dense):
    
    def __init__(self, input_shape_value, fan_in, gain, *args, **kwargs):
        #if 'kernel_initializer' in kwargs:
        #    raise Exception("Cannot override kernel_initializer")
        #super().__init__(kernel_initializer=normal(0,1), **kwargs)

        #do I need to pass the units or the initializer??
        shape = [input_shape_value, args[0]]
        if fan_in is None: fan_in = np.prod(shape[:-1])
        self.fan_in = fan_in
        self.gain = gain
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        #might need to change to tf functions
        #build with input_shape

        super().build(input_shape)

        std = self.gain / np.sqrt(self.fan_in) # He init
        self.wscale = std

    def call(self, inputs):

        #original dense call function
        # if dtype:
        #     if inputs.dtype.base_dtype != dtype.base_dtype:
        #     inputs = math_ops.cast(inputs, dtype=dtype)

        # rank = inputs.shape.rank
        # if rank == 2 or rank is None:
        #     if isinstance(inputs, sparse_tensor.SparseTensor):
        #     outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, kernel)
        #     else:
        #     outputs = gen_math_ops.mat_mul(inputs, kernel)
        # # Broadcast kernel to inputs.
        # else:
        #     outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
        #     # Reshape the output back to the original ndim of the input.
        #     if not context.executing_eagerly():
        #     shape = inputs.shape.as_list()
        #     output_shape = shape[:-1] + [kernel.shape[-1]]
        #     outputs.set_shape(output_shape)

        # if bias is not None:
        #     outputs = nn_ops.bias_add(outputs, bias)

        # if activation is not None:
        #     outputs = activation(outputs)

        # return outputs

        # Use for tensorflow 2

        return core_ops.dense(
           inputs,
           self.kernel*self.wscale,
           self.bias,
           self.activation,
           dtype=self._compute_dtype_object)


        # output = K.dot(inputs, self.kernel*self.wscale) # scale kernel
        # if self.use_bias:
        #     output = K.bias_add(output, self.bias, data_format='channels_first')
        # if self.activation is not None:
        #     output = self.activation(output)
        # return output


class Conv2DLReq(Conv2D):
    
    def __init__(self, input_shape_value, fan_in, gain, *args, **kwargs):
        #if 'kernel_initializer' in kwargs:
        #    raise Exception("Cannot override kernel_initializer")
        #super().__init__(kernel_initializer=normal(0,1), **kwargs)

        #do I need to pass the units or the initializer??
        shape = [input_shape_value, args[0]]
        if fan_in is None: fan_in = np.prod(shape[:-1])
        self.fan_in = fan_in
        self.gain = gain
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        #might need to change to tf functions
        #build with input_shape

        super().build(input_shape)
        #gain=np.sqrt(2)
        std = self.gain / np.sqrt(self.fan_in) # He init
        self.wscale = std

    def call(self, inputs):
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                self.kernel*self.wscale, # scale kernel
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        return outputs

        #original dense call function

class Lerp_clip_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Lerp_clip_layer, self).__init__()

    def call(self, inputs):
        a = inputs[0]
        b = inputs[1]
        t = 0.0 #inputs[2]
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
        


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

    def call(self, inputs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + self.epsilon)    
        


#----------------------------------------------------------------------------



def G_paper(
    latents_in,                         # First input: Latent vectors [minibatch, latent_size].                                        # Second input: Labels [minibatch, label_size].
    num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
    resolution          = 32,           # Output resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 128,          # Maximum number of feature maps in any layer.
    latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = False,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
    structure           = 'linear',  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution > 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    if latent_size is None: latent_size = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu
    
    latents_in.set_shape([None, 3, resolution, resolution])

    combo_in = tf.cast(latents_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    
    # Building blocks.
    def block(x, res, y): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            y_temp=y
            if res == 5: # 32x32
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = tf.reshape(x, [-1, 32, 8, 8])
                    x = PN(act(apply_bias(x)))
                with tf.variable_scope('Conv'):
                    x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=64, kernel=3, use_wscale=use_wscale))))
            else: #x64 and up
                if fused_scale:
                    for r in range(resolution_log2, res-1, -1):
                        with tf.variable_scope('Conv_down%d_%d' %(res, r)):
                            y_temp= act(apply_bias(conv2d_downscale2d(y_temp, fmaps=nf(res-5), kernel=3, use_wscale=use_wscale)))
                    with tf.variable_scope('Conv0_up'):
                        x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-5), kernel=3, use_wscale=use_wscale))))   
                        x = tf.concat([x, y_temp], axis=1)                     

                else:                   
                    for r in range(resolution_log2, res-1, -1):
                        with tf.variable_scope('Conv_down%d_%d' %(res, r)):
                          y_temp= act(apply_bias(conv2d_downscale2d(y_temp, fmaps=nf(res-5), kernel=3, use_wscale=use_wscale)))
                    x = upscale2d(x)
                    x = tf.concat([x, y_temp], axis=1)
                    with tf.variable_scope('ConvConcat'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-5), kernel=3, use_wscale=use_wscale))))
                    with tf.variable_scope('Conv0'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-5), kernel=3, use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-5), kernel=3, use_wscale=use_wscale))))
            return x

    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=num_channels, kernel=1, use_wscale=use_wscale)))


    def torgb(x, res): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

     # Encoder: 

    print(combo_in)
    x = fromrgb(combo_in, resolution_log2)
    y = x
    for res in range(resolution_log2, 4, -1):
        with tf.variable_scope('Conv_down%d' %res):
            x= act(apply_bias(conv2d_downscale2d(x, fmaps=128, kernel=3, use_wscale=use_wscale)))
    with tf.variable_scope('Conv_down4'):
        x= act(apply_bias(conv2d_downscale2d(x, fmaps=32, kernel=3, use_wscale=use_wscale)))    
    with tf.variable_scope('Dense0up'):                
        x = act(apply_bias(dense(x, fmaps=2048, use_wscale=use_wscale)))
        x = PN(act(apply_bias(x)))  
    with tf.variable_scope('Dense1up'):
        combo_in = act(apply_bias(dense(x, fmaps=2048, gain=1, use_wscale=use_wscale)))
 
    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, 5, y)
        images_out = torgb(x, 5)
        for res in range(6, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res, y)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, res, lod):
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            if res > 5: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(combo_in, 5, resolution_log2 - 6)               

    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    return images_out



def Generator_model(resolution=256,num_channels=3,lod_in = 0.0): #confirm params and confirm initializers strides bias
    resolution_log2 = int(np.log2(resolution))
    dshape=( 3, resolution, resolution) 
    normal_gain = np.sqrt(2)
    latent = Input(shape=dshape)
    #lod_in = tf.cast(tf.Variable(initial_value=np.float32(0.0), trainable=False), 'float32')

    fmap_base           = 8192        # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 128          # Maximum number of feature maps in any layer.

    def nf(stage): 
        print('stage ' + str(stage))
        fm = min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        print(fm)
        return fm

    # -----------------------------------------------------------------------------------------------------------------
    # Encoder ---------------------------------------------------------------------------------------------------------
    # convert from rgb ------------------------------------
    x = Conv2DLReq(latent.shape[1].value, None, normal_gain, num_channels, kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(latent)
    #x = Conv2D(3, kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(latent)
    x = LeakyReLU()(x)
    y = x

    # -----------------------------------------------------
    # loop downscale --------------------------------------
    for res in range(resolution_log2, 4, -1):
        #x = tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True
        #x = Conv2D(128, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
        x = Conv2DLReq(x.shape[1].value, None, normal_gain, 128, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(x)
        x = LeakyReLU()(x)

    # -----------------------------------------------------
    # final downscale -------------------------------------
    #x = Conv2D(32, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
    x = Conv2DLReq(x.shape[1].value, None, normal_gain, 32, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(x)
    x = LeakyReLU()(x)
    # -----------------------------------------------------
    # Dense -----------------------------------------------
    x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    x = DenseLReq(x.shape[1].value, None, normal_gain, 2048, bias_initializer='ones')(x)
    x = LeakyReLU()(x)
    # -----------------------------------------------------
    # Pixel Normalization ---------------------------------
    # add bias
    # add leakyRelu
    x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8) # Pixel Normalization
    # -----------------------------------------------------
    # Combo_in --------------------------------------------
    #x = Dense(2048, bias_initializer='ones')(x)
    x = DenseLReq(x.shape[1].value, None, normal_gain, 2048, bias_initializer='ones')(x)
    x = LeakyReLU()(x)

    # -----------------------------------------------------------------------------------------------------------------
    # Block resolution 5 ----------------------------------------------------------------------------------------------
    y_temp=y
    # Normalize -------------------------------------------
    x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8) # Pixel Normalization
    # -----------------------------------------------------
    # Dense -----------------------------------------------
    x = tf.reshape(x, [-1, 32, 8, 8])
    # add bias
    x = LeakyReLU()(x)
    x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8) # Pixel Normalization
    # -----------------------------------------------------
    # upscale conv ----------------------------------------
    x = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(x)
    # use transpose? <------
    #x = Conv2D(64, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
    x = Conv2DLReq(x.shape[1].value, None, normal_gain, 64, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
    x = LeakyReLU()(x)
    x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8) # Pixel Normalization

    # -----------------------------------------------------
    # to rgb ----------------------------------------------
    images_out = Conv2DLReq(x.shape[1].value, None, 1 , num_channels, kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
    
    #Model(inputs=[latent], outputs=[images_out]).summary()
    #return

    # -----------------------------------------------------
    # block cycle for 64 and up ---------------------------
    for res in range(6, resolution_log2 + 1):
        lod = resolution_log2 - res
        y_temp=y
        for r in range(resolution_log2, res-1, -1):
            y_temp = Conv2DLReq(y_temp.shape[1].value, None, normal_gain, nf(res-5), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(y_temp)
            y_temp = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(y_temp)
            y_temp = LeakyReLU()(y_temp)

        x = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(x)
        x = tf.concat([x, y_temp], axis=1)

        x = Conv2DLReq(x.shape[1].value, None, normal_gain, nf(res-5), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
        x = LeakyReLU()(x)
        x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8) # Pixel Normalization

        x = Conv2DLReq(x.shape[1].value, None, normal_gain, nf(res-5), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
        x = LeakyReLU()(x)
        x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8) # Pixel Normalization

        x = Conv2DLReq(x.shape[1].value, None, normal_gain, nf(res-5), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
        x = LeakyReLU()(x)
        x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8) # Pixel Normalization
    
        img = Conv2DLReq(x.shape[1].value, None, 1 , num_channels, kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)

        images_out = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(images_out)

        images_out = lerp_clip(img, images_out, lod_in - lod)

    Model(inputs=[latent], outputs=[images_out]).summary()
    
    return Model(inputs=[latent], outputs=[images_out])
 


def named_generator_model(resolution=256,num_channels=3): #confirm params and confirm initializers strides bias
    resolution_log2 = int(np.log2(resolution))
    dshape=( 3, resolution, resolution) 
    normal_gain = np.sqrt(2)
    latent = Input(shape=dshape)
    #lod_in = Input(shape=(1,))
    lod_in= Input(shape=(1), batch_size=1, name='lod_in')
    #lod_in = tf.cast(tf.Variable(initial_value=np.float32(0.0), trainable=False), 'float32')

    fmap_base           = 8192        # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 128          # Maximum number of feature maps in any layer.

    def nf(stage): 
        #print('stage ' + str(stage))
        fm = min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        #print(fm)
        return fm

    print(latent.shape[1])

    # -----------------------------------------------------------------------------------------------------------------
    # Encoder ---------------------------------------------------------------------------------------------------------
    # convert from rgb ------------------------------------
    x = Conv2DLReq(latent.shape[1], None, normal_gain, num_channels, kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/FromRGB_lod' % (2**resolution_log2, 2**resolution_log2))(latent)
    #x = Conv2D(3, kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(latent)
    x = LeakyReLU()(x)
    y = x

    # -----------------------------------------------------
    # loop downscale --------------------------------------
    for res in range(resolution_log2, 4, -1):
        #x = tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True
        #x = Conv2D(128, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
        x = Conv2DLReq(x.shape[1], None, normal_gain, 128, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones',name='%dx%d/Conv_down%d' % (2**resolution_log2, 2**resolution_log2, res))(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(x)
        x = LeakyReLU()(x)

    # -----------------------------------------------------
    # final downscale -------------------------------------
    #x = Conv2D(32, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
    x = Conv2DLReq(x.shape[1], None, normal_gain, 32, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='Conv_down4')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(x)
    x = LeakyReLU()(x)
    # -----------------------------------------------------
    # Dense -----------------------------------------------
    x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
    print(x.shape)
    x = DenseLReq(x.shape[1], None, normal_gain, 2048, bias_initializer='ones', name='Dense0up')(x)
    x = LeakyReLU()(x)
    # -----------------------------------------------------
    # Pixel Normalization ---------------------------------
    # add bias
    # add leakyRelu
    x = Pixel_norm_layer(epsilon=1e-8)(x)
    #x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8) # Pixel Normalization
    # -----------------------------------------------------
    # Combo_in --------------------------------------------
    #x = Dense(2048, bias_initializer='ones')(x)
    x = DenseLReq(x.shape[1], None, normal_gain, 2048, bias_initializer='ones', name='Dense1up')(x)
    x = LeakyReLU()(x)

    # -----------------------------------------------------------------------------------------------------------------
    # Block resolution 5 ----------------------------------------------------------------------------------------------
    y_temp=y
    # Normalize -------------------------------------------
    x = Pixel_norm_layer(epsilon=1e-8)(x) # Pixel Normalization
    # -----------------------------------------------------
    # Dense -----------------------------------------------
    #tf.variable_scope('Dense'):
    x = tf.reshape(x, [-1, 32, 8, 8])
    # add bias
    x = LeakyReLU()(x)
    x = Pixel_norm_layer(epsilon=1e-8)(x) # Pixel Normalization
    # -----------------------------------------------------
    # upscale conv ----------------------------------------
    x = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(x)
    # use transpose? <------
    #x = Conv2D(64, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
    x = Conv2DLReq(x.shape[1], None, normal_gain, 64, kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/Conv' % (2**5, 2**5))(x)
    x = LeakyReLU()(x)
    x = Pixel_norm_layer(epsilon=1e-8)(x) # Pixel Normalization

    # -----------------------------------------------------
    # to rgb ----------------------------------------------
    images_out = Conv2DLReq(x.shape[1], None, 1 , num_channels, kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/ToRGB_lod' % (2**5, 2**5))(x)
    
    #Model(inputs=[latent], outputs=[images_out]).summary()
    #return

    # -----------------------------------------------------
    # block cycle for 64 and up ---------------------------
    for res in range(6, resolution_log2 + 1):
        lod = resolution_log2 - res
        y_temp=y
        for r in range(resolution_log2, res-1, -1):
            y_temp = Conv2DLReq(y_temp.shape[1], None, normal_gain, nf(res-5), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/Conv_down%d_%d' %(2**res, 2**res, res, r))(y_temp)
            y_temp = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(y_temp)
            y_temp = LeakyReLU()(y_temp)

        x = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(x)
        x = tf.concat([x, y_temp], axis=1)

        x = Conv2DLReq(x.shape[1], None, normal_gain, nf(res-5), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/ConvConcat'% (2**res, 2**res))(x)
        x = LeakyReLU()(x)
        x = Pixel_norm_layer(epsilon=1e-8)(x) # Pixel Normalization

        x = Conv2DLReq(x.shape[1], None, normal_gain, nf(res-5), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/Conv0'% (2**res, 2**res))(x)
        x = LeakyReLU()(x)
        x = Pixel_norm_layer(epsilon=1e-8)(x) # Pixel Normalization

        x = Conv2DLReq(x.shape[1], None, normal_gain, nf(res-5), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/Conv1'% (2**res, 2**res))(x)
        x = LeakyReLU()(x)
        x = Pixel_norm_layer(epsilon=1e-8)(x) # Pixel Normalization
    
        img = Conv2DLReq(x.shape[1], None, 1 , num_channels, kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/ToRGB_lod' % (2**res, 2**res))(x)

        images_out = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(images_out)

        images_out = lerp_clip(img, images_out, lod_in - lod)

    images_out = tf.identity(images_out, name='images_out')

    Model(inputs=[latent, lod_in], outputs=[images_out]).summary()
    
    return Model(inputs=[latent, lod_in], outputs=[images_out])


#----------------------------------------------------------------------------
# Discriminator network used in the paper.

def D_paper(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                    x = downscale2d(x)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
            return x
    
    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out


def Discriminator_model(resolution=256,num_channels=3, lod_in = 0.0, label_size = 0, mbstd_group_size = 4): #confirm params and confirm initializers strides bias
    resolution_log2 = int(np.log2(resolution))
    dshape=( num_channels, resolution, resolution) 
    normal_gain = np.sqrt(2)
    images_in = Input(shape=dshape)
    #lod_in = tf.cast(tf.Variable(initial_value=np.float32(0.0), trainable=False), 'float32')

    #TODO: Final 2 dense layers too sharp: add one or more dense layers in the middle


    fmap_base           = 8192         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512          # Maximum number of feature maps in any layer.
    dtype               = 'float32'

    def nf(stage): 
        fm = min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        return fm


    img = images_in
    # convert from rgb ------------------------------------
    x = Conv2DLReq(img.shape[1].value, None, normal_gain, nf(resolution_log2-1), kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='fromrgb_0')(img)
    x = LeakyReLU()(x)

    i = 0

    for res in range(resolution_log2, 2, -1):
        lod = resolution_log2 - res
        if res >= 3: # 8x8 and up
            x = Conv2DLReq(x.shape[1].value, None, normal_gain, nf(res-1), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
            x = LeakyReLU()(x)
            x = Conv2DLReq(x.shape[1].value, None, normal_gain, nf(res-2), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
            x = LeakyReLU()(x)
            x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(x)
        else:
            if mbstd_group_size > 1:
                x = minibatch_stddev_layer(x, mbstd_group_size) # <-------
            x = Conv2DLReq(x.shape[1].value, None, normal_gain, nf(res-1), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
            x = LeakyReLU()(x)
            x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
            x = DenseLReq(x.shape[1].value, None, normal_gain, nf(res-2), bias_initializer='ones')(x)
            x = LeakyReLU()(x)
            x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
            x = DenseLReq(x.shape[1].value, None, 1, 1+label_size, bias_initializer='ones')(x)

        img = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(img)
        #x = LeakyReLU()(x)
        y = Conv2DLReq(img.shape[1].value, None, normal_gain, nf(res-1-1), kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='fromrgb_'+str(res))(img)
        y = LeakyReLU()(y)

        x = lerp_clip(x, y, lod_in - lod)


    if mbstd_group_size > 1:
        x = minibatch_stddev_layer(x, mbstd_group_size) # <-------
    x = Conv2DLReq(x.shape[1].value, None, normal_gain, nf(2-1), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones')(x)
    x = LeakyReLU()(x)
    x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    x = DenseLReq(x.shape[1].value, None, normal_gain, nf(2-2), bias_initializer='ones')(x)
    x = LeakyReLU()(x)
    x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    x = DenseLReq(x.shape[1].value, None, 1, 1+label_size, bias_initializer='ones')(x)
    combo_out = x


    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')



    Model(inputs=[images_in], outputs=[scores_out, labels_out]).summary()
    
    return Model(inputs=[images_in], outputs=[scores_out, labels_out])

def named_discriminator(resolution=256, num_channels=3, label_size = 0, mbstd_group_size = 4, old_res=128):
    resolution_log2 = int(np.log2(resolution))
    dshape=( num_channels, resolution, resolution) 
    normal_gain = np.sqrt(2)
    images_in = Input(shape=dshape)
    lod_in= Input(shape=(1), batch_size=1, name='lod_in')
    #lod_in = tf.squeeze(lod_in_input, axis=0)
    #lod_in = lod_in_input
    #print(lod_in)
    #lod_in = tf.cast(tf.Variable(initial_value=np.float32(0.0), trainable=False), 'float32')

    #lod_in = tf.Variable(initial_value=0.0, trainable=False)
    #lod = tf.Variable(initial_value=0.0, trainable=False)

    #TODO: Final 2 dense layers too sharp: add one or more dense layers in the middle


    fmap_base           = 8192         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512          # Maximum number of feature maps in any layer.
    dtype               = 'float32'

    def nf(stage): 
        fm = min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        return fm

    img = images_in
    # convert from rgb ------------------------------------
    #x = Conv2DLReq(img.shape[1].value, None, normal_gain, nf(resolution_log2-1), kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='FromRGB_lod%d' % (resolution_log2 - resolution_log2))(img)
    x = Conv2DLReq(img.shape[1], None, normal_gain, nf(resolution_log2-1), kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/FromRGB_lod' % (2**resolution_log2, 2**resolution_log2))(img)
    x = LeakyReLU()(x)

    i = 0

    for res in range(resolution_log2, 2, -1):
        lod = resolution_log2 - res
        #lod.assign(resolution_log2 - res)
        if res >= 3: # 8x8 and up
            x = Conv2DLReq(x.shape[1], None, normal_gain, nf(res-1), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/Conv0' % (2**res, 2**res))(x)
            x = LeakyReLU()(x)
            x = Conv2DLReq(x.shape[1], None, normal_gain, nf(res-2), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/Conv1' % (2**res, 2**res))(x)
            x = LeakyReLU()(x)
            x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(x)
        else:
            if mbstd_group_size > 1:
                x = Minibatch_stddev_layer(mbstd_group_size)(x) # <-------
            x = Conv2DLReq(x.shape[1], None, normal_gain, nf(res-1), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones',name='%dx%d/Conv' % (2**res, 2**res))(x)
            x = LeakyReLU()(x)
            x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
            x = DenseLReq(x.shape[1], None, normal_gain, nf(res-2), bias_initializer='ones', name='%dx%d/Dense0' % (2**res, 2**res))(x)
            x = LeakyReLU()(x)
            x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
            x = DenseLReq(x.shape[1], None, 1, 1+label_size, bias_initializer='ones', name='%dx%d/Dense1' % (2**res, 2**res))(x)

        img = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(img)
        #x = LeakyReLU()(x)
        y = Conv2DLReq(img.shape[1], None, normal_gain, nf(res-1-1), kernel_size=1, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/FromRGB_lod' % (2**(res - 1), 2**(res - 1)))(img)
        y = LeakyReLU()(y)


        #x = Lerp_clip_layer()([x, y])
        x = lerp_clip(x, y, lod_in - lod)


    if mbstd_group_size > 1:
        x = Minibatch_stddev_layer(mbstd_group_size)(x) # <-------
    x = Conv2DLReq(x.shape[1], None, normal_gain, nf(2-1), kernel_size=3, strides=(1,1), data_format='channels_first',padding='same', bias_initializer='ones', name='%dx%d/Conv' % (4, 4))(x)
    x = LeakyReLU()(x)
    x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
    x = DenseLReq(x.shape[1], None, normal_gain, nf(2-2), bias_initializer='ones', name='%dx%d/Dense0' % (4, 4))(x)
    x = LeakyReLU()(x)
    x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
    x = DenseLReq(x.shape[1], None, 1, 1+label_size, bias_initializer='ones', name='%dx%d/Dense1' % (4, 4))(x)
    combo_out = x


    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')



    Model(inputs=[images_in, lod_in], outputs=[scores_out, labels_out]).summary()
    
    return Model(inputs=[images_in, lod_in], outputs=[scores_out, labels_out])

