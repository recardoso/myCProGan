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

from myCProGan.layers import *

#Use for tensorflow 2
#from tensorflow.python.keras.layers.ops import core as core_ops

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

def generator(resolution=256,num_channels=3,num_replicas=1): #confirm params and confirm initializers strides bias

    resolution_log2 = int(np.log2(resolution))
    noise_shape = 2048 #shape of the latent noise
    ae_shape = 2048 #shape of the latent from the autoencoder
    #dshape=( 3, resolution, resolution) 
    latent = Input(shape=noise_shape, name='latent')
    latent = tf.cast(latent, tf.float32)
    ae_latent = Input(shape=3*ae_shape, name='ae_latent')
    ae_latent = tf.cast(ae_latent, tf.float32)
    lod_in = Input(shape=(), batch_size=num_replicas, name='lod_in')
    lod_in = tf.cast(lod_in, tf.float32)
    dtype = 'float32'

    #x,y = encoder(latent, resolution_log2, gain=np.sqrt(2))

    #concatenate latent noise with varaiational encoder latent

    x = tf.concat([latent, ae_latent], axis=1) #axis 0 or 1? {is 0 the batch or not?}

    #transform input to a dense (pixelnorm first?)
    # transfprmation to dense might remove the random noise making the whole training pointless
    #x = transform_generator_input(x,5,resolution_log2,gain=np.sqrt(2),norm=True)


    x = generator_block(x, 2, resolution_log2)
    images_out = torgb(x, 2, filters=num_channels)
    for res in range(3, resolution_log2):
        lod = resolution_log2 - res
        x = generator_block(x, res, resolution_log2)
        img = torgb(x, res, filters=num_channels)
        images_out = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(images_out)
        images_out = Lerp_clip_layer()([img, images_out, lod_in - lod])
        #lerp_clip(img, images_out, lod_in - lod)

    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    
    Model(inputs=[latent, ae_latent, lod_in], outputs=[images_out]).summary()
    
    return Model(inputs=[latent, ae_latent, lod_in], outputs=[images_out])

def discriminator(resolution=256, num_channels=3, label_size = 0, mbstd_group_size = 4, old_res=128, num_replicas=1):
    resolution_log2 = int(np.log2(resolution))
    dshape=( num_channels, resolution, resolution) 
    normal_gain = np.sqrt(2)
    images_in = Input(shape=dshape)
    images_in = tf.cast(images_in, tf.float32)
    lod_in= Input(shape=(), batch_size=num_replicas, name='lod_in')
    lod_in = tf.cast(lod_in, tf.float32)

    fmap_base           = 8192         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512          # Maximum number of feature maps in any layer.

    #TODO: Final 2 dense layers too sharp: add one or more dense layers in the middle
    dtype               = 'float32'

    img = images_in
    x = fromrgb(img, resolution_log2, filters=nf(resolution_log2-1, fmap_base, fmap_decay, fmap_max))
    for res in range(resolution_log2, 2, -1):
        lod = resolution_log2 - res
        x = discriminator_block(x, res, resolution_log2, mbstd_group_size = mbstd_group_size)
        img = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_first')(img)
        y = fromrgb(img, res - 1, filters=nf(res - 1-1, fmap_base, fmap_decay, fmap_max))
        x = Lerp_clip_layer()([x, y, lod_in - lod])
    combo_out = discriminator_block(x, 2, resolution_log2, mbstd_group_size = mbstd_group_size)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    #labels_out = tf.identity(combo_out[:, 1:], name='labels_out')

    Model(inputs=[images_in, lod_in], outputs=[scores_out]).summary()
    
    return Model(inputs=[images_in, lod_in], outputs=[scores_out])

class Combined_Discriminator(tf.keras.Model):
    """Combined Discriminator with Local and GLobal Discriminator"""

    def __init__(self, resolution=256, num_channels=3, label_size = 0, mbstd_group_size = 4, old_res=128, num_replicas=1):
        super(Combined_Discriminator, self).__init__()
        #self.latent_dim = latent_dim

        self.local_discriminator = discriminator(resolution=resolution//2, num_channels=num_channels, label_size = label_size, mbstd_group_size = mbstd_group_size, old_res=old_res, num_replicas=num_replicas)

        self.global_discriminator = discriminator(resolution=resolution, num_channels=num_channels, label_size = label_size, mbstd_group_size = mbstd_group_size, old_res=old_res, num_replicas=num_replicas)


    def use_local_discriminator(self, image, lod, training_flag=True):
        scores = self.local_discriminator([image, lod], training=training_flag)
        return scores

    #calculate z
    def use_global_discriminator(self, image, lod, training_flag=True):
        scores = self.global_discriminator([image, lod], training=training_flag)
        return scores


def Variational_encoder(resolution=256,num_channels=3,latent_dim=128,kernel_size=3,base_filter=32, variational=False):
    #variational encoder to encode the 3 images from the original image
    #dshape=( num_channels, int(resolution/2), int(resolution/2*3))
    dshape=( num_channels, int(resolution/2), int(resolution/2))
    images_in = Input(shape=dshape)

    resolution_log2 = int(np.log2(resolution))

    x = images_in

    #some conv2d for the rgb convertion?
    x = Conv2D(filters=num_channels, kernel_size=3, strides=(1, 1),  padding='same', data_format='channels_first')(x)

    for n_layer in range(0,resolution_log2-3):
        #if base_filter*2**n_layer > 128:
            #filters = 128
        #else:
        filters = base_filter*2**n_layer
        name = 'Conv_' + str(filters)
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(2, 2), padding='same', data_format='channels_first', name=name)(x)
        name = 'BatchNorm_' + str(filters)
        x = BatchNormalization(axis=1,name=name)(x)
        x = LeakyReLU(0.2)(x)
    flat = Flatten()(x)
    # No activation
    #one for mean and another for logvar
    #mean = Dense(latent_dim)(x)
    #logvar = Dense(latent_dim)(x)

    #mean = tf.identity(mean, name='encoded_mean')
    #logvar = tf.identity(logvar, name='encoded_logvar')

    if variational:
        Model(inputs=[images_in], outputs=[flat]).summary()
    
        return Model(inputs=[images_in], outputs=[flat])

    else:
        name = 'Dense_latent'
        latent = Dense(latent_dim, name=name)(flat)
        latent = tf.identity(latent, name='encoded_latent')
        Model(inputs=[images_in], outputs=[latent]).summary()
        
        return Model(inputs=[images_in], outputs=[latent])

    #Model(inputs=[images_in], outputs=[mean, logvar]).summary()
    
    #return Model(inputs=[images_in], outputs=[mean, logvar])

    # Model(inputs=[images_in], outputs=[latent]).summary()
    
    # return Model(inputs=[images_in], outputs=[latent])

def Variational_decoder(resolution=256,num_channels=3,latent_dim=128,kernel_size=3,base_filter=32,variational=False):
    latent_in = Input(shape=(latent_dim))
    resolution_log2 = int(np.log2(resolution))

    base_units = base_filter*2**(resolution_log2-3-1)

    #values 2 and 6 are in accordance with the concatenated images 
    # 4 corresponds to 256
    # 2 corresponds to 128
    # 6 corresponds to 384

    # x = Dense(units=base_units*4*12)(latent_in)
    # x = Reshape(target_shape=(base_units, 4, 12))(x)

    name = 'Dense_latent'
    x = Dense(units=base_units*4*4, name=name)(latent_in)
    x = Reshape(target_shape=(base_units, 4, 4))(x)

    for n_layer in range(resolution_log2-3-1,-1,-1):
        #if base_filter*2**n_layer > 128:
        #    filters = 128
        #else:
        filters = base_filter*2**n_layer
        name = 'Conv_' + str(filters)
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding='same', data_format='channels_first', name=name)(x)
        name = 'BatchNorm_' + str(filters)
        x = BatchNormalization(axis=1,name=name)(x)
        x = LeakyReLU(0.2)(x)
    # No activation

    #to RGB
    #x = Conv2DTranspose(filters=3, kernel_size=1, strides=1,data_format='channels_first',activation='sigmoid')(x)
    x = Conv2D(filters=num_channels, kernel_size=3, strides=1,data_format='channels_first',padding='same')(x)

    decoded_image = tf.identity(x, name='decoded_image')

    Model(inputs=[latent_in], outputs=[decoded_image]).summary()
    
    return Model(inputs=[latent_in], outputs=[decoded_image])


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, resolution=256, num_channels=3, latent_dim=128,kernel_size=3,base_filter=32):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = Variational_encoder(resolution=resolution,num_channels=num_channels,latent_dim=self.latent_dim,kernel_size=kernel_size,base_filter=base_filter)

        self.decoder = Variational_decoder(resolution=resolution,num_channels=num_channels,latent_dim=self.latent_dim,kernel_size=kernel_size,base_filter=base_filter)


    def encode(self, x, training_flag=True):
        latent = self.encoder(x, training=training_flag)
        return latent
    #calculate z
    def reparameterize(self, mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False, apply_tahn=False):
        logits = self.decoder(z, training=True)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        if apply_tahn:
            probs = tf.keras.activations.tanh(logits)
            return probs
        return logits

    #eps should be z?
    @tf.function
    def sample(self,eps=None,latent_dim=128):
        if eps is None:
            eps = tf.random.normal(shape=(100, latent_dim))
        return self.decode(eps,self.decoder, apply_sigmoid=True)

class Beta_VAE(tf.keras.Model):
    def __init__(self, resolution=256, num_channels=3, latent_dim=128,kernel_size=3,base_filter=32,beta=4,gamma=10.,max_capacity=25.,Capacity_max_iter=10000, anneal_steps= 200,alpha=1.):
        super(Beta_VAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.C_max = max_capacity
        self.C_stop_iter = Capacity_max_iter
        self.anneal_steps = anneal_steps
        self.alpha = alpha

        self.encoder = Variational_encoder(resolution=resolution,num_channels=num_channels,latent_dim=self.latent_dim,kernel_size=kernel_size,base_filter=base_filter, variational=True)

        self.mean = Dense(latent_dim)
        self.var = Dense(latent_dim)

        self.decoder = Variational_decoder(resolution=resolution,num_channels=num_channels,latent_dim=self.latent_dim,kernel_size=kernel_size,base_filter=base_filter, variational=True)

    def encode(self, x):
        latent = self.encoder(x, training=True)
        mean = self.mean(latent)
        var = self.var(latent)
        return mean,var

    def decode(self, z, apply_sigmoid=False, apply_tahn=False):
        logits = self.decoder(z, training=True)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        if apply_tahn:
            probs = tf.keras.activations.tanh(logits)
            return probs
        return logits

    def reparameterize(self, mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return eps * tf.exp(logvar * .5) + mean


if __name__ == "__main__":
    a = np.zeros((1, 3, 256, 256))
    size = 128
    a1= a[:,:, :(size),:(size)]
    a2= a[:,:, (size):,:(size)]
    a3= a[:,:, :(size),(size):]
    a4= a[:,:, :(size), :(size)]
    aleft = tf.concat([a1, a2], axis=3)
    aall3 = tf.concat([aleft, a3], axis=3)
    print(tf.shape(aall3))
    Variational_encoder(resolution=256, base_filter=8)
    Variational_decoder(resolution=256, base_filter=8)