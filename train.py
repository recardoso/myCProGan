import os
import time
import numpy as np
import tensorflow as tf
import glob
import argparse
import json
import datetime
import os


#import config
#import tfutil
#import dataset
import myCProGan.networks2 as networks2
import myCProGan.loss as loss
from myCProGan.snapshots import *
from myCProGan.dataset import *
#import misc

from tensorflow.keras.optimizers import Adam

#from tensorflow.keras.utils import to_categorical, plot_model
#import tensorflow_datasets as tfds

from PIL import Image
#import click


import pickle
#import optuna





def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def lerp(a, b, t):
    return a + (b - a) * t


#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    x = tf.cast(x, tf.float32)
    x = adjust_dynamic_range(x, drange_data, drange_net)

    if mirror_augment:
        s = tf.shape(x)
        mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
        mask = tf.tile(mask, [1, s[1], s[2], s[3]])
        x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
    # Smooth crossfade between consecutive levels-of-detail.
    s = tf.shape(x)
    y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
    y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
    y = tf.tile(y, [1, 1, 1, 2, 1, 2])
    y = tf.reshape(y, [-1, s[1], s[2], s[3]])
    x = lerp(x, y, lod - tf.floor(lod))

    # Upscale to match the expected input/output size of the networks. 256x256 only useful if we are not changing the model
    # s = tf.shape(x)
    # factor = tf.cast(2 ** tf.floor(lod), tf.int32)
    # x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    # x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    # x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    # tf.print(tf.shape(x))
    return x

def process_reals_cvae(x, drange_data, drange_net):
    x = tf.cast(x, tf.float32)

#     x = tf.clip_by_value(x, 50, 150)
#     x =  (x - 50)*(((250-0)//(150-50))+0)

#     hist = tf.histogram_fixed_width(x, [0,256], nbins=256)
#     tf.math.cumsum(hist)
#     h = np.round((r_cdf - r_cdf[0]) / (256*256 - r_cdf[0]) * 255)
#     r_test_image[i] = h[el]


    x = adjust_dynamic_range(x, drange_data, drange_net)

    return x


#----------------------------------------------------------------------------
# Training chedule (lod updates)

def TrainingSchedule(
    cur_nimg,                           #current number of images
    dataset_res_log2,                   #log2 of the resolution of the dataset
    num_gpus,                           #number of gpus
    lod_initial_resolution  = 4,        # Image resolution used at the beginning. (I think this is related to the initial resolution of the network, 4 in this case)
    lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
    minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
    minibatch_dict          = {},       # Resolution-specific overrides.
    max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
    G_lrate_base            = 0.002,    # Learning rate for the generator.
    G_lrate_dict            = {},       # Resolution-specific overrides.
    D_lrate_base            = 0.001,    # Learning rate for the discriminator.
    D_lrate_dict            = {},       # Resolution-specific overrides.
    tick_kimg_base          = 160,      # Default interval of progress snapshots.
    tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:20, 1024:10}): # Resolution-specific overrides.

    # Training phase.
    kimg = cur_nimg / 1000.0
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    lod = dataset_res_log2
    lod -= np.floor(np.log2(lod_initial_resolution))
    lod -= phase_idx
    if lod_transition_kimg > 0:
        lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
    lod = max(lod, 0.0)
    resolution = 2 ** (dataset_res_log2 - int(np.floor(lod)))

    # Minibatch size.
    minibatch = minibatch_dict.get(resolution, minibatch_base)
    minibatch -= minibatch % num_gpus
    if resolution in max_minibatch_per_gpu:
        minibatch = min(minibatch, max_minibatch_per_gpu[resolution] * num_gpus)

    # Other parameters.
    G_lrate = G_lrate_dict.get(resolution, G_lrate_base)
    D_lrate = D_lrate_dict.get(resolution, D_lrate_base)
    tick_kimg = tick_kimg_dict.get(resolution, tick_kimg_base)

    lod = np.float32(lod)
    return lod, resolution, G_lrate, D_lrate


#----------------------------------------------------------------------------
# Training Cycle


#for 4 GPUS
# lod_training_kimg = 300
# lod_transition_kimg = 1500

# minibatch_base = 16
# minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
# max_minibatch_per_gpu = {256: 8, 512: 4, 1024: 2}
# G_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}
# D_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}
# total_kimg = 12000

#for 1 GPU
# lod_training_kimg = 300
# lod_transition_kimg = 1500

# minibatch_base = 4
# minibatch_dict = {4: 128, 8: 64, 16: 32, 32: 16, 64: 8, 128: 4}
# max_minibatch_per_gpu = {256: 2, 512: 2, 1024: 2}
# G_lrate_dict = {256: 0.0015}
# D_lrate_dict = {256: 0.0015}
# total_kimg = 12000

# @click.command()
# @click.option('--multi_node', default=False)
# #@click.option('--workers', nargs='+', default=[]) #use like this --workers 10.1.10.58:12345 10.1.10.250:12345
# @click.option('--index', type=int, default=0)
# @click.option('--use_gs', default=False)
# @click.option('--datapath', default='') #'/eos/user/r/redacost/progan/tfrecords1024/'
# @click.option('--outpath', default='')
# @click.option('--init_res', type=int, default=32, help='initial resolution of the progressive gan')
# @click.option('--max_res', type=int, default=256, help='maximum resolution based on the dataset')
# @click.option('--change_model', default=False, help='if the model is increased based on resolution')
# @click.option('--lod_training_kimg', type=int, default=300)
# @click.option('--lod_transition_kimg', type=int, default=1500)
# @click.option('--minibatch_base', type=int, default=32)
# @click.option('--total_kimg', type=int, default=12000)
# @click.option('--minibatch_repeats', type=int, default=4)
# @click.option('--D_repeats', type=int, default=2)
# @click.option('--batch_size', type=int, default=8)
# @click.option('--load', default=False)
def train_cycle(multi_node=False,index=0,use_gs=False,datapath='',outpath='',init_res=32,max_res=256,change_model=False,lod_training_kimg=300,lod_transition_kimg=1500,minibatch_base=32,total_kimg=12000,minibatch_repeats=4,D_repeats=2,batch_size=8,load=False,use_gpus=True):
    #initialization of variables
    BETA_1 = 0.0
    BETA_2 = 0.99
    EPSILON = 1e-8

    dict_batch_size = batch_size

    max_res_log2 = int(np.log2(max_res))
    num_replicas = 1
    curr_res_log2 = int(np.log2(init_res))
    prev_res = -1

    minibatch_dict = {4: 1024, 8: 512, 16: 256, 32: dict_batch_size*8, 64: dict_batch_size*4, 128: dict_batch_size*2}
    max_minibatch_per_gpu = {256: dict_batch_size, 512: 8, 1024: 8}
    G_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}
    D_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}


    #training repeats
    curr_image = 0

    #initialization of paths
    #tfrecord_dir = '/home/renato/dataset'
    tfrecord_dir_train = datapath + '/train-floods' #'/home/renato/dataset/satimages'
    tfrecord_dir_test = datapath + '/test-floods' #'/home/renato/dataset/satimages'
    
    print(tfrecord_dir_train)
    print(tfrecord_dir_test)

    start_init = time.time()
    f = [0.9, 0.1] # train, test fractions might be necessary

    #net_size
    net_size = init_res
    
    #create logs for tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = '/home/jovyan/runs/train' #'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = '/home/jovyan/runs/test' #'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)


    #start strategy
    if multi_node:
        
        workers=["10.1.10.58:12345", "10.1.10.250:12345"]

        print(multi_node)
        print(workers)
        print(index)

        #tf_config
        os.environ["TF_CONFIG"] = json.dumps({
            'cluster': {'worker': workers},#["10.1.10.58:12345", "10.1.10.250:12345"]},
            'task': {'type': 'worker', 'index': index}
        })
    if use_gpus:
        if not multi_node:
            strategy = tf.distribute.MirroredStrategy()
            use_stategy = True
        if multi_node:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)
    #elif use_tpu:
    else:
        use_stategy = False


    #initialize models and optimizer
    if use_stategy:
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        num_replicas = strategy.num_replicas_in_sync
        if change_model:
            with strategy.scope():
                gen = networks2.generator(init_res, num_channels=1, num_replicas = num_replicas) #choose resolution
                #plot_model(gen, show_shapes=True, dpi=64)
                disc = networks2.Combined_Discriminator(init_res, num_channels=1, num_replicas = num_replicas) #choose resolution
                #plot_model(disc, show_shapes=True, dpi=64)
                cvae = networks2.CVAE(resolution=init_res, base_filter=int(32*(256/init_res)),latent_dim=2048, num_channels=1)
                cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
                cvae.load_weights(outpath+'saved_models/vae/cvae_models/cvae_Final_'+str(init_res)+'.h5')
        else:
            with strategy.scope():
                gen = networks2.generator(256, num_replicas = num_replicas)
                #gen = networks2.generator(256, num_replicas = num_replicas) #choose resolution
                #plot_model(gen, show_shapes=True, dpi=64)
                disc = networks2.Combined_Discriminator(256, num_replicas = num_replicas) #choose resolution
                #plot_model(disc, show_shapes=True, dpi=64)
                cvae = networks2.CVAE(resolution=256, base_filter=32,latent_dim=512)
                cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
                cvae.load_weights(outpath+'saved_models/vae/cvae_models/cvae_Final.h5')
    else:
        if change_model:
            gen = networks2.generator(init_res,  num_channels=1) #choose resolution
            #plot_model(gen, show_shapes=True, dpi=64)
            disc = networks2.Combined_Discriminator(init_res, num_channels=1) #choose resolution
            #plot_model(disc, show_shapes=True, dpi=64)
            cvae = networks2.CVAE(resolution=init_res, base_filter=int(32*(256/init_res)),latent_dim=2048,  num_channels=1)
            cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
            cvae.load_weights(outpath+'saved_models/vae/cvae_models/cvae_Final_'+str(init_res)+'.h5')
        else:
            gen = networks2.generator(256) #choose resolution
            #plot_model(gen, show_shapes=True, dpi=64)
            disc = networks2.Combined_Discriminator(256) #choose resolution
            #plot_model(disc, show_shapes=True, dpi=64)
            cvae = networks2.CVAE(resolution=256, base_filter=32,latent_dim=512)
            cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
            cvae.load_weights(outpath+'saved_models/vae/cvae_models/cvae_Final.h5')

    #D_optimizer = Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
    #G_optimizer = Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)

    #load train data
    tfr_files = sorted(glob.glob(os.path.join(tfrecord_dir_train, '*.tfrecords')))
    dataset_list, batch_sizes_list = get_dataset(tfr_files, flood=True, num_gpus=num_replicas, minibatch_base = minibatch_base, minibatch_dict = minibatch_dict, max_minibatch_per_gpu = max_minibatch_per_gpu) #all the images for all the lods
    if use_stategy:
        distributed_dataset_train = {}
        for key, value in dataset_list.items():
            distributed_dataset_train[key] = strategy.experimental_distribute_dataset(value)
    else:
        distributed_dataset_train = {}
        for key, value in dataset_list.items():
            distributed_dataset_train[key] = value
            
    #load test data        
    tfr_files = sorted(glob.glob(os.path.join(tfrecord_dir_test, '*.tfrecords')))
    dataset_list, batch_sizes_list = get_dataset(tfr_files, flood=True, num_gpus=num_replicas, minibatch_base = minibatch_base, minibatch_dict = minibatch_dict, max_minibatch_per_gpu = max_minibatch_per_gpu) #all the images for all the lods
    if use_stategy:
        distributed_dataset_test = {}
        for key, value in dataset_list.items():
            distributed_dataset_test[key] = strategy.experimental_distribute_dataset(value)
    else:
        distributed_dataset_test = {}
        for key, value in dataset_list.items():
            distributed_dataset_test[key] = value

    #do any necessary data processing

    # Start training
    epoch_start = time.time()

    def training_step(train_batch, test_batch, lod_in_value):

        #lod_in = tf.constant([[lod_in_value]])
        #lod_in = tf.constant(lod_in_value, shape=(num_replicas, 1))
        lod_in = lod_in_value
        mirror_augment = False
        drange_net = [-1,1]
        drange_data = [0, 255]

        train_batch = process_reals(train_batch, lod_in_value, mirror_augment, drange_data, drange_net)
        test_batch = process_reals(test_batch, lod_in_value, mirror_augment, drange_data, drange_net)
        #return batch

        gen_loss = loss.Generator_loss(gen, disc, cvae, train_batch, batch_size, G_optimizer, lod_in=lod_in, training_set=None, cond_weight = 1.0, network_size=net_size, global_batch_size = global_batch_size)
        
        disc_loss, global_loss, local_loss = loss.combined_Discriminator_loss(gen, disc, cvae, train_batch, batch_size, D_optimizer, lod_in=lod_in, training_set=None, labels=None, wgan_lambda = 10.0, wgan_epsilon = 0.001, wgan_target = 1.0, cond_weight = 1.0,  network_size=net_size, global_batch_size = global_batch_size)  

        #gen_loss = loss.original_Generator_loss(gen, disc, batch, batch_size, G_optimizer, lod_in=lod_in, training_set=None, cond_weight = 1.0, network_size=net_size, global_batch_size = global_batch_size)
        #disc_loss = loss.original_Discriminator_loss(gen, disc, batch, batch_size, D_optimizer, lod_in=lod_in, training_set=None, labels=None, wgan_lambda = 10.0, wgan_epsilon = 0.001, wgan_target = 1.0, cond_weight = 1.0,  network_size=net_size, global_batch_size = global_batch_size)
        
        gen_test_loss = loss.Generator_test_loss(gen, disc, cvae, test_batch, batch_size, G_optimizer, lod_in=lod_in, training_set=None, cond_weight = 1.0, network_size=net_size, global_batch_size = global_batch_size)
        
        disc_test_loss, global_test_loss, local_test_loss = loss.combined_Discriminator_test_loss(gen, disc, cvae, test_batch, batch_size, D_optimizer, lod_in=lod_in, training_set=None, labels=None, wgan_lambda = 10.0, wgan_epsilon = 0.001, wgan_target = 1.0, cond_weight = 1.0,  network_size=net_size, global_batch_size = global_batch_size)  
        
        train_loss(gen_loss)
        test_loss(gen_test_loss)



        #return batch,batch,batch,batch
        return gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss

    if use_stategy:
        @tf.function
        def training_step_tf_fuction_32(train_dataset, test_dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = strategy.run(training_step, args=(next(train_dataset), next(test_dataset), lod_in_value))

            gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
            disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_loss, axis=None)
            global_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_loss, axis=None)
            local_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_loss, axis=None)
            
            gen_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss, axis=None)
            disc_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss, axis=None)
            global_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_test_loss, axis=None)
            local_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_test_loss, axis=None)

            return gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss

        @tf.function
        def training_step_tf_fuction_64(train_dataset, test_dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = strategy.run(training_step, args=(next(train_dataset), next(test_dataset), lod_in_value))

            gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
            disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_loss, axis=None)
            global_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_loss, axis=None)
            local_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_loss, axis=None)
            
            gen_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss, axis=None)
            disc_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss, axis=None)
            global_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_test_loss, axis=None)
            local_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_test_loss, axis=None)

            return gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss

        @tf.function
        def training_step_tf_fuction_128(train_dataset, test_dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = strategy.run(training_step, args=(next(train_dataset), next(test_dataset), lod_in_value))

            gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
            disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_loss, axis=None)
            global_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_loss, axis=None)
            local_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_loss, axis=None)
            
            gen_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss, axis=None)
            disc_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss, axis=None)
            global_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_test_loss, axis=None)
            local_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_test_loss, axis=None)

            return gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss

        @tf.function
        def training_step_tf_fuction_256(train_dataset, test_dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = strategy.run(training_step, args=(next(train_dataset), next(test_dataset), lod_in_value))

            gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
            disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_loss, axis=None)
            global_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_loss, axis=None)
            local_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_loss, axis=None)
            
            gen_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss, axis=None)
            disc_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss, axis=None)
            global_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_test_loss, axis=None)
            local_test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_test_loss, axis=None)

            return gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss
    else:
        @tf.function
        def training_step_tf_fuction_32(train_dataset, test_dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = training_step(next(train_dataset), next(test_dataset), lod_in_value)
            return gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss

        @tf.function
        def training_step_tf_fuction_64(train_dataset, test_dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = training_step(next(train_dataset), next(test_dataset), lod_in_value)
            return gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss

        @tf.function
        def training_step_tf_fuction_128(train_dataset, test_dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = training_step(next(train_dataset), next(test_dataset), lod_in_value)
            return gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss

        @tf.function
        def training_step_tf_fuction_256(train_dataset, test_dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = training_step(next(train_dataset), next(test_dataset), lod_in_value)
            return gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss

    train_time = time.time()      

    gen_loss_train = []
    disc_loss_train = []
    global_loss_train = [] 
    local_loss_train = []

    gen_loss_test = []
    disc_loss_test = []
    global_loss_test = [] 
    local_loss_test = []

    
    if not change_model:
        gw, gh, reals, fakes, grid = setup_image_grid(datapath+'/floods', [3,256,256],  m_size = '1080p', is_ae=False)
    else:
        num_colors = 1
        gw, gh, reals, fakes, grid = setup_image_grid(datapath+'/floods', [num_colors,init_res,init_res],  m_size = '1080p', is_ae=False)

    # curr_image = 4512000
    # with strategy.scope():
    #     #gen.save_weights('generator.h5')
    #     #gen = networks.named_generator_model(resolution)
    #     gen.load_weights('saved_models/generator_4512.h5', by_name=True)

    #     #disc.save_weights('discriminator.h5')
    #     #disc = networks.named_discriminator(resolution)
    #     disc.load_weights('saved_models/discriminator_4512.h5', by_name=True)

    def optimizer_weight_setting(model, optimizer,optimizer_weights):
        grad_vars = model.trainable_variables
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        optimizer.apply_gradients(zip(zero_grads, grad_vars))
        #with open(optimizer_weights, 'rb') as f:
        weight_values = np.load(optimizer_weights, allow_pickle=True)
        optimizer.set_weights(weight_values)

        return True

    while curr_image < total_kimg * 1000:
        # update model / variables (lr, lod, dataset) if needed 
        # print(curr_image)


        if load:
            print('Loading...')
            load_current_image = 2240
            load_resolution = 64
            curr_image = load_current_image * 1000
            prev_res= -1
            resolution = load_resolution


            if use_stategy:
                with strategy.scope():
                    #increase models
                    #gen.save_weights('generator.h5')
                    gen = networks2.generator(resolution, num_channels=1, num_replicas = num_replicas)
                    gen.load_weights('saved_models/pgan/models/generator_'+str(load_current_image)+'.h5', by_name=True)
                    #os.remove('generator.h5') 

                    #disc.save_weights('discriminator.h5')
                    disc = networks2.Combined_Discriminator(resolution, num_channels=1, num_replicas = num_replicas)
                    disc.built = True
                    disc.load_weights('saved_models/pgan/models/discriminator_'+str(load_current_image)+'.h5', by_name=True)
                    #os.remove('discriminator.h5') 

                    cvae = networks2.CVAE(resolution=resolution, base_filter=int(32*(256/resolution)),latent_dim=2048,num_channels=1)
                    cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
                    cvae.load_weights(outpath+'saved_models/vae/cvae_models/cvae_Final_'+str(resolution)+'.h5')

                    gw, gh, reals, fakes, grid = setup_image_grid(datapath+'/floods', [1,resolution,resolution],  m_size = '1080p', is_ae=False)
                    




        lod, resolution, G_lrate, D_lrate = TrainingSchedule(
        curr_image,                           
        max_res_log2,                   
        num_replicas,                           
        lod_initial_resolution  = init_res,        
        lod_training_kimg       = lod_training_kimg,      
        lod_transition_kimg     = lod_transition_kimg,      
        minibatch_base          = minibatch_base,       
        minibatch_dict          = minibatch_dict,       
        max_minibatch_per_gpu   = max_minibatch_per_gpu,       
        G_lrate_dict            = G_lrate_dict,       
        D_lrate_dict            = D_lrate_dict)  

        lod = lod.item()
        lod_in_value = tf.constant(lod)

        if prev_res != resolution:
            print('Increase Resoltion from %d to %d' % (prev_res, resolution), flush=True)

            #change dataset

            #lod is 6 for 4x4 and 0 for 256x256 (Should be change to match lod obtainned from TrainingSchedule)
            resolution_log2 = int(np.log2(resolution))
            lod_dataset = max_res_log2 - resolution_log2
            #get dataset and new batch size
            if use_stategy:
                dataset_train = distributed_dataset_train[lod_dataset]
                dataset_test = distributed_dataset_test[lod_dataset]
            else: 
                dataset_train = distributed_dataset_train[lod_dataset]
                dataset_test = distributed_dataset_test[lod_dataset]
                
            dataset_iter_train = iter(dataset_train)
            dataset_iter_test = iter(dataset_test)
            
            
            batch_size = int(batch_sizes_list[lod_dataset] / num_replicas)
            global_batch_size = batch_sizes_list[lod_dataset]
            
            lod_in_value = tf.constant(lod)

            print(lod_in_value)

            if change_model:
                net_size = int(resolution)
                print(net_size)
            else:
                net_size = int(resolution * 2**tf.math.floor(lod_in_value))
                print(net_size)

            #change optmizer
            if use_stategy:
                print('Changing optimizer')
                print(load)
                with strategy.scope():
                    D_optimizer = Adam(learning_rate=G_lrate, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
                    G_optimizer = Adam(learning_rate=D_lrate, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
                    
                    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
                    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)


                print('Changing optimizer')
                print(load)
                if load:
                    print('Loading Changing optimizer')
                    #load_current_image = 2240
                    g_opt_updt = strategy.run(optimizer_weight_setting, args=(gen, G_optimizer, outpath+'saved_models/pgan_optimizers/g_optimizer_'+str(load_current_image)+'.npy'))
                    d_opt_updt = strategy.run(optimizer_weight_setting, args=(disc, D_optimizer,outpath+'saved_models/pgan_optimizers/d_optimizer_'+str(load_current_image)+'.npy'))
                    load = False
                    print('Loaded Optimizer')
            else:
                print('Changing optimizer non strategy', flush=True)
                D_optimizer = Adam(learning_rate=G_lrate, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
                G_optimizer = Adam(learning_rate=D_lrate, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
                train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
                test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
                if load == True:
                    load_current_image = 1920
                    optimizer_weight_setting(gen, G_optimizer, outpath+'saved_models/pgan_optimizers/g_optimizer_'+str(load_current_image)+'.npy')
                    optimizer_weight_setting(disc, D_optimizer,outpath+'saved_models/pgan_optimizers/d_optimizer_'+str(load_current_image)+'.npy')
                    load = False
            

            #change model
            if prev_res != -1 and change_model:
                if use_stategy:
                    with strategy.scope():
                        #increase models
                        gen.save_weights('generator.h5')
                        gen = networks2.generator(resolution, num_replicas = num_replicas,num_channels=1)
                        gen.load_weights('generator.h5', by_name=True)
                        os.remove('generator.h5') 

                        disc.save_weights('discriminator.h5')
                        disc = networks2.Combined_Discriminator(resolution, num_replicas = num_replicas, num_channels=1)
                        disc.built = True
                        disc.load_weights('discriminator.h5', by_name=True)
                        os.remove('discriminator.h5') 

                        cvae = networks2.CVAE(resolution=resolution, base_filter=int(32*(256/resolution)),latent_dim=2048,num_channels=1)
                        cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
                        cvae.load_weights(outpath+'saved_models/vae/cvae_models/cvae_Final_'+str(resolution)+'.h5')

                        gw, gh, reals, fakes, grid = setup_image_grid(datapath+'/floods', [1,resolution,resolution],  m_size = '1080p', is_ae=False)
                else:
                    gen.save_weights('generator.h5')
                    gen = networks2.generator(resolution, num_replicas = num_replicas,num_channels=1)
                    gen.load_weights('generator.h5', by_name=True)
                    os.remove('generator.h5') 

                    disc.save_weights('discriminator.h5')
                    disc = networks2.Combined_Discriminator(resolution, num_replicas = num_replicas, num_channels=1)
                    disc.built = True
                    disc.load_weights('discriminator.h5', by_name=True)
                    os.remove('discriminator.h5') 

                    cvae = networks2.CVAE(resolution=resolution, base_filter=int(32*(256/resolution)),latent_dim=2048,num_channels=1)
                    cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
                    cvae.load_weights(outpath+'saved_models/vae/cvae_models/cvae_Final_'+str(resolution)+'.h5')

                    gw, gh, reals, fakes, grid = setup_image_grid(datapath+'/floods', [1,resolution,resolution],m_size='1080p',is_ae=False)
            
            elif prev_res != -1 and not change_model:
                if use_stategy:
                    with strategy.scope():
                        gen.save_weights('generator.h5')
                        gen = networks2.generator(256, num_replicas = num_replicas)
                        gen.load_weights('generator.h5', by_name=True)
                        os.remove('generator.h5') 

                        disc.save_weights('discriminator.h5')
                        disc = networks2.Combined_Discriminator(256, num_replicas = num_replicas)
                        disc.built = True
                        disc.load_weights('discriminator.h5', by_name=True)
                        os.remove('discriminator.h5') 

            #optimizer learning rate (and reset?)

            prev_res = resolution

            print('Finished changing resolution', flush=True)

        # Run training ops.
        for repeat in range(minibatch_repeats):
            #print(repeat)
            if resolution == 32:
                gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = training_step_tf_fuction_32(dataset_iter_train, dataset_iter_test, lod_in_value)
            elif resolution == 64:
                gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = training_step_tf_fuction_64(dataset_iter_train, dataset_iter_test, lod_in_value)
            elif resolution == 128:
                gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = training_step_tf_fuction_128(dataset_iter_train, dataset_iter_test, lod_in_value)
            elif resolution == 256:
                gen_loss, disc_loss, global_loss, local_loss, gen_test_loss, disc_test_loss, global_test_loss, local_test_loss = training_step_tf_fuction_256(dataset_iter_train, dataset_iter_test, lod_in_value)
            #print(lossval.numpy())

        #Run Test


        # get stats and save model
#         gen_loss_train.append(gen_loss.numpy())
#         disc_loss_train.append(disc_loss.numpy())
#         global_loss_train.append(global_loss.numpy()) 
#         local_loss_train.append(local_loss.numpy())
        
#         gen_loss_test.append(gen_test_loss.numpy())
#         disc_loss_test.append(disc_test_loss.numpy())
#         global_loss_test.append(global_test_loss.numpy()) 
#         local_loss_test.append(local_test_loss.numpy())

        if curr_image % (global_batch_size * 100) == 0:
            print(curr_image)
            print(time.time() - train_time )
            print(lod_in_value)
            # get stats
            train_time = time.time()
            print ("-------------------------------------------------------------------------------------------------------------", flush=True)
            print ("{:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(' ', 'Generator', 'Discriminator', 'Global', 'Local'))
            print ("{:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format('Train', gen_loss.numpy(), disc_loss.numpy(), global_loss.numpy(), local_loss.numpy()))
            print ("{:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format('Test', gen_test_loss.numpy(), disc_test_loss.numpy(), global_test_loss.numpy(), local_test_loss.numpy()))
            print ("-------------------------------------------------------------------------------------------------------------",flush=True)
#             

        if curr_image % (global_batch_size * 100) == 0:
            if change_model:
                grid = construct_grid_to_save_pgan(gw, gh, reals, grid, cvae_model=cvae, gen_model=gen, lod_in=lod_in_value,size=resolution)
            else:
                grid = construct_grid_to_save_pgan(gw, gh, reals, grid, cvae_model=cvae, gen_model=gen, lod_in=lod_in_value,size=256)
            save_grid_pgan(gw, gh,grid, step=curr_image,outpath=outpath)

        if curr_image % (global_batch_size * 10000) == 0:
            # save model
            gen.save_weights(outpath+'saved_models/pgan/models/generator_'+str(int(curr_image/1000))+'.h5')
            disc.save_weights(outpath+'saved_models/pgan/models/discriminator_'+str(int(curr_image/1000))+'.h5')
#             pickle.dump({'gen_loss': gen_loss_train, 'disc_loss': disc_loss_train, 'global_loss': global_loss_train, 'local_loss': local_loss_train, 'gen_loss_test': gen_loss_test, 'disc_loss_test': disc_loss_test, 'global_loss_test': global_loss_test, 'local_loss_test': local_loss_test}, open(outpath+'losses.pkl', 'wb'))
            # save optimizeer
            np.save(outpath+'saved_models/pgan_optimizers/g_optimizer_'+str(int(curr_image/1000))+'.npy', G_optimizer.get_weights())
            np.save(outpath+'saved_models/pgan_optimizers/d_optimizer_'+str(int(curr_image/1000))+'.npy', D_optimizer.get_weights())
            print('Model Saved', flush=True)
          
        if curr_image % (global_batch_size * 100) == 0:
            #write tensorboard logs
            tensor_board_step = curr_image // (global_batch_size * 100)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=tensor_board_step)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=tensor_board_step)

            template = 'Image {}, Loss: {}, Test Loss: {}'
            print(template.format(tensor_board_step, train_loss.result(), test_loss.result()))

            # Reset metrics every epoch
            train_loss.reset_states()
            test_loss.reset_states()
      


        curr_image += global_batch_size

    gen.save_weights(outpath+'generator_Final.h5')
    disc.save_weights(outpath+'discriminator_Final.h5')
    return



def encoder_train_cycle(lr=0.00005):
    #initialization of variables
    #BETA_1 = 0.0
    #BETA_2 = 0.99
    EPSILON = 1e-8
    LR = lr

    num_replicas = 1

    list_res = [32,64,128,256]
    
    use_gpus = True
    use_multi_gpus = False
 

    #training repeats
    n_epochs = 150

    #initialization of paths
    #tfrecord_dir = '/home/renato/dataset'
    #tfrecord_dir = '/home/renato/dataset/satimages'
    #tfrecord_dir =  '/eos/user/r/redacost/progan/tfrecords1024/'
    #tfrecord_dir = '/eos/user/r/redacost/progan/Balanced_dataset/'
    tfrecord_dir = '/eos/user/r/redacost/progan/floods/'

    start_init = time.time()
    f = [0.9, 0.1] # train, test fractions might be necessary

    
    minibatch_base = 32
    minibatch_dict = {4: 1024, 8: 512, 16: 256, 32: 256, 64: 128, 128: 128}
    max_minibatch_per_gpu = {256: 128, 512: 8, 1024: 8}



    #start strategy
    if use_gpus:
        strategy = tf.distribute.MirroredStrategy()
        use_stategy = True
    elif use_multi_gpus:
        strategy = tf.distribute.MirroredStrategy()
        use_stategy = True
    #elif use_tpu:
    else:
        use_stategy = False

    if use_stategy:
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        num_replicas = strategy.num_replicas_in_sync
        # global_batch_size = 32 * num_replicas
        # print('Global batch size: ' + str(global_batch_size))


    #load data

    #eurosat
    # DATA_DIR = '/home/renato/dataset/eurosatimages'
    # (train, val, test) = tfds.load("eurosat/rgb", split=["train[:100%]", "train[80%:90%]", "train[90%:]"])

    # def prepare_training_data(datapoint):
    #     input_image = datapoint["image"]
    #     data = tf.transpose(input_image, [2,0,1])
    #     return data

    # dset = train.map(prepare_training_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #UNOSAT
    #count = sum(1 for _ in tf.data.TFRecordDataset(tfr_file))
    #print(count)

    

    # Start training
    epoch_start = time.time()

    def training_step(batch, iterat):
        mirror_augment = False
        drange_net = [-1,1]
        drange_data = [0, 255]

        batch = process_reals_cvae(batch, drange_data, drange_net)
        #return batch

        #reconstruction_loss, kl_loss = loss.wasserstein_auto_encoder_loss(cvae, batch, global_batch_size, opt, kernel_type = 'imq')
        reconstruction_loss, kl_loss = loss.wasserstein_auto_encoder_loss(cvae, batch, global_batch_size, opt, net_size, kernel_type = 'rbf')
        #reconstruction_loss, kl_loss = loss.Beta_TC_auto_encoder_loss(cvae, batch, global_batch_size, opt, iterat, desintangled=False)


        return  reconstruction_loss, kl_loss

    if use_stategy:
        @tf.function
        def training_step_tf_fuction_4(dataset,iterat):
            reconstruction_loss, kl_loss = strategy.run(training_step, args=(next(dataset),iterat))

            reconstruction_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, reconstruction_loss, axis=None)
            kl_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, kl_loss, axis=None)
            #rgb_angle = strategy.reduce(tf.distribute.ReduceOp.SUM, rgb_angle, axis=None)

            return reconstruction_loss, kl_loss
        @tf.function
        def training_step_tf_fuction_8(dataset,iterat):
            reconstruction_loss, kl_loss = strategy.run(training_step, args=(next(dataset),iterat))

            reconstruction_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, reconstruction_loss, axis=None)
            kl_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, kl_loss, axis=None)
            #rgb_angle = strategy.reduce(tf.distribute.ReduceOp.SUM, rgb_angle, axis=None)

            return reconstruction_loss, kl_loss
        @tf.function
        def training_step_tf_fuction_16(dataset,iterat):
            reconstruction_loss, kl_loss = strategy.run(training_step, args=(next(dataset),iterat))

            reconstruction_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, reconstruction_loss, axis=None)
            kl_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, kl_loss, axis=None)
            #rgb_angle = strategy.reduce(tf.distribute.ReduceOp.SUM, rgb_angle, axis=None)

            return reconstruction_loss, kl_loss
        @tf.function
        def training_step_tf_fuction_32(dataset,iterat):
            reconstruction_loss, kl_loss = strategy.run(training_step, args=(next(dataset),iterat))

            reconstruction_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, reconstruction_loss, axis=None)
            kl_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, kl_loss, axis=None)
            #rgb_angle = strategy.reduce(tf.distribute.ReduceOp.SUM, rgb_angle, axis=None)

            return reconstruction_loss, kl_loss
        @tf.function
        def training_step_tf_fuction_64(dataset,iterat):
            reconstruction_loss, kl_loss = strategy.run(training_step, args=(next(dataset),iterat))

            reconstruction_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, reconstruction_loss, axis=None)
            kl_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, kl_loss, axis=None)
            #rgb_angle = strategy.reduce(tf.distribute.ReduceOp.SUM, rgb_angle, axis=None)

            return reconstruction_loss, kl_loss
        @tf.function
        def training_step_tf_fuction_128(dataset,iterat):
            reconstruction_loss, kl_loss = strategy.run(training_step, args=(next(dataset),iterat))

            reconstruction_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, reconstruction_loss, axis=None)
            kl_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, kl_loss, axis=None)
            #rgb_angle = strategy.reduce(tf.distribute.ReduceOp.SUM, rgb_angle, axis=None)

            return reconstruction_loss, kl_loss
        @tf.function
        def training_step_tf_fuction_256(dataset,iterat):
            reconstruction_loss, kl_loss = strategy.run(training_step, args=(next(dataset),iterat))

            reconstruction_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, reconstruction_loss, axis=None)
            kl_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, kl_loss, axis=None)
            #rgb_angle = strategy.reduce(tf.distribute.ReduceOp.SUM, rgb_angle, axis=None)

            return reconstruction_loss, kl_loss
    else:
        @tf.function
        def training_step_tf_fuction_32(dataset):
            reconstruction_loss, kl_loss = training_step(next(dataset),)
            return reconstruction_loss, kl_loss



    tfr_files = sorted(glob.glob(os.path.join(tfrecord_dir, '*.tfrecords')))
    dataset_list, batch_sizes_list = get_dataset(tfr_files, flood=True, num_gpus=num_replicas, minibatch_base = minibatch_base, minibatch_dict = minibatch_dict, max_minibatch_per_gpu = max_minibatch_per_gpu) #all the images for all the lods
    if use_stategy:
        distributed_dataset = {}
        for key, value in dataset_list.items():
            distributed_dataset[key] = strategy.experimental_distribute_dataset(value)
            
#     dset = retrieve_balanced_tfrecords(tfr_files, 32 * num_replicas)
#     dataset = strategy.experimental_distribute_dataset(dset)

        
    train_time = time.time()      

    reconstruction_loss_train = []
    kl_loss_train = []

    prev_res = -1


    for el in list_res:

        resolution_log2 = int(np.log2(el))
        lod_dataset = int(np.log2(256)) - resolution_log2
        #get dataset and new batch size
        if use_stategy:
            dataset = distributed_dataset[lod_dataset]
        else: 
            dataset = dataset_list[lod_dataset]
        dataset_iter = iter(dataset)
        batch_size = int(batch_sizes_list[lod_dataset] / num_replicas)
        global_batch_size = batch_sizes_list[lod_dataset]
#         global_batch_size = 32 * num_replicas
#         batch_size = int(global_batch_size / num_replicas)

        
        net_size = int(el / 2)
        print(net_size)

        dataset_size = 62747 
        batch_repeats = int(dataset_size / global_batch_size)

        print('N_batches = ' + str(batch_repeats))

        #setup image grid
        gw, gh, reals, fakes, grid = setup_image_grid('/eos/user/r/redacost/progan/floods/',[1,net_size,net_size],  m_size = '1080p')
        print('image grid done')


        if prev_res == -1:
            #initialize models and optimizer
            if use_stategy:
                with strategy.scope():
                    #cvae = networks2.CVAE(resolution=256, base_filter=32,latent_dim=512)
                    cvae = networks2.CVAE(resolution=el, base_filter=int(32*(256/el)),latent_dim=2048,num_channels=1) #change value 32 according to init res
                    #cvae = networks2.Beta_VAE(resolution=el, base_filter=int(32*(256/el)),latent_dim=512)
            else:
                cvae = networks2.CVAE(resolution=el, base_filter=int(32*(256/el)),latent_dim=2048,num_channels=1)
            prev_res=0
        #change model
        else:
            with strategy.scope():
                #increase models
                cvae.save_weights('cvae.h5')
                cvae = networks2.CVAE(resolution=el, base_filter=int(32*(256/el)),latent_dim=2048,num_channels=1) #change value 32 according to init res
                cvae.built = True
                cvae.load_weights('cvae.h5', by_name=True)
                os.remove('cvae.h5') 



        with strategy.scope():
            opt = Adam(learning_rate=LR, epsilon=EPSILON)

        n_iterat = 0

        for epoch in range(n_epochs):
            print('Epoch: ' + str(epoch))

            train_time = time.time() 

            iterat = tf.constant(n_iterat)
            iterat = tf.cast(iterat, tf.float32)

            # Run training ops.
            for _ in range(batch_repeats):
                if el == 4:
                    reconstruction_loss, kl_loss = training_step_tf_fuction_4(dataset_iter,iterat)
                elif el == 8:
                    reconstruction_loss, kl_loss = training_step_tf_fuction_8(dataset_iter,iterat)
                elif el == 16:
                    reconstruction_loss, kl_loss = training_step_tf_fuction_16(dataset_iter,iterat)
                elif el == 32:
                    reconstruction_loss, kl_loss = training_step_tf_fuction_32(dataset_iter,iterat)
                elif el == 64:
                    reconstruction_loss, kl_loss = training_step_tf_fuction_64(dataset_iter,iterat)
                elif el == 128:
                    reconstruction_loss, kl_loss = training_step_tf_fuction_128(dataset_iter,iterat)
                elif el == 256:
                    reconstruction_loss, kl_loss = training_step_tf_fuction_256(dataset_iter,iterat)

                #Run Test

                reconstruction_loss_train.append(reconstruction_loss.numpy())
                kl_loss_train.append(kl_loss.numpy())

            n_iterat +=1
            
            # print(time.time() - train_time )
            # print('Reconstruction Loss ' + str(reconstruction_loss.numpy()))
            # print('Kl Loss ' + str(kl_loss.numpy()))

            # save model
            if epoch % 10 == 0:
                cvae.save_weights('saved_models/vae/cvae_'+str(int(epoch))+'.h5')
                grid = construct_grid_to_save(gw, gh, reals, fakes, grid, model=cvae, step=epoch,size=net_size)
                save_grid(gw, gh,grid, step=epoch)
                # print('Model Saved')
                print(time.time() - train_time )
                print('Reconstruction Loss ' + str(reconstruction_loss.numpy()))
                print('Kl Loss ' + str(kl_loss.numpy()))
                #print('rgb_angle ' + str(rgb_angle.numpy())) 
            pickle.dump({'reconstruction_loss': reconstruction_loss_train, 'kl_loss': kl_loss_train}, open('saved_models/vae/losses_'+str(lr)+'.pkl', 'wb'))
            

        cvae.save_weights('saved_models/vae/cvae_models/cvae_Final_'+str(el)+'.h5')
        grid = construct_grid_to_save(gw, gh, reals, fakes, grid, model=cvae, step=epoch,size=net_size)
        save_grid(gw, gh,grid, step=epoch)

    total_loss = reconstruction_loss + kl_loss

    return cvae #total_loss



# if __name__ == "__main__":
# #     os.environ["CUDA_VISIBLE_DEVICES"]="0"
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)

#     # For pgan training
#     train_cycle()

#     #for auto encoder training
#     # model = encoder_train_cycle(lr=8.5e-5)


#     #study = optuna.create_study()
#     #study.optimize(objective, n_trials=100, callbacks=[print_best_callback])
#     print('Finished Training')
    
# #     datapath = '/eos/user/r/redacost/progan/floods/'
# #     tfrecord_dir = datapath
# #     tfr_file = datapath + 'floods-r08.tfrecords'
# #     num_threads     = 1

# #     dset = tf.data.TFRecordDataset(tfr_file)
# #     dset = dset.map(parse_tfrecord_tf_floods, num_parallel_calls=num_threads)

# #     dataset_iter = iter(dset)

# #     n_images = 62747
# #     for i in range(n_images):
# #         if i%100 == 0:
# #             print(i)
# #         img = next(dataset_iter).numpy()[0][0]

# #         #plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# #         #plt.show()
