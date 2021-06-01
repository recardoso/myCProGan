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
    normal_gain = np.sqrt(2)
    latent = Input(shape=dshape)
    latent = tf.cast(latent, tf.float32)
    lod_in= Input(shape=(), batch_size=num_replicas, name='lod_in')
    lod_in = tf.cast(lod_in, tf.float32)


    fmap_base           = 8192        # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0         # log2 feature map reduction when doubling the resolution.
    fmap_max            = 128         # Maximum number of feature maps in any layer.

    def nf(stage): 
        fm = min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        return fm

