import os
import time
import numpy as np
import tensorflow as tf
import glob
import argparse
import json

#import config
#import tfutil
#import dataset
import networks2
import loss
from snapshots import *
#import misc

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical, plot_model
#import tensorflow_datasets as tfds

from PIL import Image

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

    # Upscale to match the expected input/output size of the networks. 256x256
    s = tf.shape(x)
    factor = tf.cast(2 ** tf.floor(lod), tf.int32)
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    #tf.print(tf.shape(x))
    return x

def process_reals_cvae(x, drange_data, drange_net):
    x = tf.cast(x, tf.float32)
    x = adjust_dynamic_range(x, drange_data, drange_net)

    return x

#----------------------------------------------------------------------------
# Dataset Reading and Processing

def parse_tfrecord_tf(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
    #data = tf.io.decode_raw(features['data'], tf.uint8)
    data = tf.io.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

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
        dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=buffer_mb<<20)
        #use parse function
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

    print('Dataset Read')

    return _tf_datasets, _tf_batch_sizes

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


def train_cycle():
    #initialization of variables
    BETA_1 = 0.0
    BETA_2 = 0.99
    EPSILON = 1e-8

    parser = get_parser()
    params = parser.parse_args()

    multi_node = params.multi_node
    use_gs = params.use_gs
    datapath = params.datapath
    outpath = params.outpath

    init_res = params.init_res
    max_res = params.max_res
    change_model = params.change_model
    max_res_log2 = int(np.log2(max_res))
    num_replicas = 1
    curr_res_log2 = int(np.log2(init_res))
    prev_res = -1

    lod_training_kimg = params.lod_training_kimg
    lod_transition_kimg = params.lod_transition_kimg
    minibatch_base = params.minibatch_base 
    minibatch_dict = {4: 1024, 8: 512, 16: 256, 32: 8, 64: 8, 128: 8}
    max_minibatch_per_gpu = {256: 8, 512: 8, 1024: 8}
    G_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}
    D_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}
    total_kimg = params.total_kimg

    #training repeats
    minibatch_repeats = params.minibatch_repeats
    D_repeats = params.D_repeats
    
    use_gpus = True


    #training repeats
    curr_image = 0

    #initialization of paths
    #tfrecord_dir = '/home/renato/dataset'
    tfrecord_dir = datapath #'/home/renato/dataset/satimages'

    start_init = time.time()
    f = [0.9, 0.1] # train, test fractions might be necessary

    #net_size
    net_size = init_res

    #start strategy
    if multi_node:
        workers = params.workers
        index = params.index

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
                gen = networks2.generator(init_res, num_replicas = num_replicas) #choose resolution
                #plot_model(gen, show_shapes=True, dpi=64)
                disc = networks2.Combined_Discriminator(init_res, num_replicas = num_replicas) #choose resolution
                #plot_model(disc, show_shapes=True, dpi=64)
                cvae = networks2.CVAE(resolution=256, base_filter=32,latent_dim=512)
                cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
                cvae.load_weights('saved_models/vae/cvae_models/cvae_Final.h5')
        else:
            with strategy.scope():
                gen = networks2.generator(256, num_replicas = num_replicas)
                #gen = networks2.generator(256, num_replicas = num_replicas) #choose resolution
                #plot_model(gen, show_shapes=True, dpi=64)
                disc = networks2.Combined_Discriminator(256, num_replicas = num_replicas) #choose resolution
                #plot_model(disc, show_shapes=True, dpi=64)
                cvae = networks2.CVAE(resolution=256, base_filter=32,latent_dim=512)
                cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
                cvae.load_weights('saved_models/vae/cvae_models/cvae_Final.h5')
    else:
        if change_model:
            gen = networks2.generator(init_res) #choose resolution
            #plot_model(gen, show_shapes=True, dpi=64)
            disc = networks2.Combined_Discriminator(init_res) #choose resolution
            #plot_model(disc, show_shapes=True, dpi=64)
            cvae = networks2.CVAE(resolution=256, base_filter=32,latent_dim=512)
            cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
            cvae.load_weights('saved_models/vae/cvae_models/cvae_Final.h5')
        else:
            gen = networks2.generator(256) #choose resolution
            #plot_model(gen, show_shapes=True, dpi=64)
            disc = networks2.Combined_Discriminator(256) #choose resolution
            #plot_model(disc, show_shapes=True, dpi=64)
            cvae = networks2.CVAE(resolution=256, base_filter=32,latent_dim=512)
            cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
            cvae.load_weights('saved_models/vae/cvae_models/cvae_Final.h5')

    #D_optimizer = Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
    #G_optimizer = Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)

    #load data
    tfr_files = sorted(glob.glob(os.path.join(tfrecord_dir, '*.tfrecords')))
    dataset_list, batch_sizes_list = get_dataset(tfr_files, num_gpus=num_replicas, minibatch_base = minibatch_base, minibatch_dict = minibatch_dict, max_minibatch_per_gpu = max_minibatch_per_gpu) #all the images for all the lods
    if use_stategy:
        distributed_dataset = {}
        for key, value in dataset_list.items():
            distributed_dataset[key] = strategy.experimental_distribute_dataset(value)

    #do any necessary data processing

    # Start training
    epoch_start = time.time()

    def training_step(batch, lod_in_value):

        #lod_in = tf.constant([[lod_in_value]])
        #lod_in = tf.constant(lod_in_value, shape=(num_replicas, 1))
        lod_in = lod_in_value
        mirror_augment = False
        drange_net = [-1,1]
        drange_data = [0, 255]

        batch = process_reals(batch, lod_in_value, mirror_augment, drange_data, drange_net)
        #return batch

        gen_loss = loss.Generator_loss(gen, disc, cvae, batch, batch_size, G_optimizer, lod_in=lod_in, training_set=None, cond_weight = 1.0, network_size=net_size, global_batch_size = global_batch_size)
        disc_loss, global_loss, local_loss = loss.combined_Discriminator_loss(gen, disc, cvae, batch, batch_size, D_optimizer, lod_in=lod_in, training_set=None, labels=None, wgan_lambda = 10.0, wgan_epsilon = 0.001, wgan_target = 1.0, cond_weight = 1.0,  network_size=net_size, global_batch_size = global_batch_size)  

        #gen_loss = loss.original_Generator_loss(gen, disc, batch, batch_size, G_optimizer, lod_in=lod_in, training_set=None, cond_weight = 1.0, network_size=net_size, global_batch_size = global_batch_size)
        #disc_loss = loss.original_Discriminator_loss(gen, disc, batch, batch_size, D_optimizer, lod_in=lod_in, training_set=None, labels=None, wgan_lambda = 10.0, wgan_epsilon = 0.001, wgan_target = 1.0, cond_weight = 1.0,  network_size=net_size, global_batch_size = global_batch_size)

        return gen_loss, disc_loss, global_loss, local_loss

    if use_stategy:
        @tf.function
        def training_step_tf_fuction_32(dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss = strategy.run(training_step, args=(next(dataset), lod_in_value))

            gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
            disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_loss, axis=None)
            global_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_loss, axis=None)
            local_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_loss, axis=None)

            return gen_loss, disc_loss, global_loss, local_loss

        @tf.function
        def training_step_tf_fuction_64(dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss = strategy.run(training_step, args=(next(dataset), lod_in_value))

            gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
            disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_loss, axis=None)
            global_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_loss, axis=None)
            local_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_loss, axis=None)

            return gen_loss, disc_loss, global_loss, local_loss

        @tf.function
        def training_step_tf_fuction_128(dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss = strategy.run(training_step, args=(next(dataset), lod_in_value))

            gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
            disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_loss, axis=None)
            global_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_loss, axis=None)
            local_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_loss, axis=None)

            return gen_loss, disc_loss, global_loss, local_loss

        @tf.function
        def training_step_tf_fuction_256(dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss = strategy.run(training_step, args=(next(dataset), lod_in_value))

            gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
            disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_loss, axis=None)
            global_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, global_loss, axis=None)
            local_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, local_loss, axis=None)

            return gen_loss, disc_loss, global_loss, local_loss
    else:
        @tf.function
        def training_step_tf_fuction_32(dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss = training_step(next(dataset), lod_in_value)
            return gen_loss, disc_loss, global_loss, local_loss

        @tf.function
        def training_step_tf_fuction_64(dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss = training_step(next(dataset), lod_in_value)
            return gen_loss, disc_loss, global_loss, local_loss

        @tf.function
        def training_step_tf_fuction_128(dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss = training_step(next(dataset), lod_in_value)
            return gen_loss, disc_loss, global_loss, local_loss

        @tf.function
        def training_step_tf_fuction_256(dataset, lod_in_value):
            gen_loss, disc_loss, global_loss, local_loss = training_step(next(dataset), lod_in_value)
            return gen_loss, disc_loss, global_loss, local_loss

    train_time = time.time()      

    gen_loss_train = []
    disc_loss_train = []
    global_loss_train = [] 
    local_loss_train = []


    gw, gh, reals, fakes, grid = setup_image_grid([3,256,256],  m_size = '1080p', is_ae=False)

    # curr_image = 4512000
    # with strategy.scope():
    #     #gen.save_weights('generator.h5')
    #     #gen = networks.named_generator_model(resolution)
    #     gen.load_weights('saved_models/generator_4512.h5', by_name=True)

    #     #disc.save_weights('discriminator.h5')
    #     #disc = networks.named_discriminator(resolution)
    #     disc.load_weights('saved_models/discriminator_4512.h5', by_name=True)


    while curr_image < total_kimg * 1000:
        # update model / variables (lr, lod, dataset) if needed 
        print(curr_image)


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
            print('Increase Resoltion from %d to %d' % (prev_res, resolution))

            #change dataset

            #lod is 6 for 4x4 and 0 for 256x256 (Should be change to match lod obtainned from TrainingSchedule)
            resolution_log2 = int(np.log2(resolution))
            lod_dataset = max_res_log2 - resolution_log2
            #get dataset and new batch size
            if use_stategy:
                dataset = distributed_dataset[lod_dataset]
            else: 
                dataset = dataset_list[lod_dataset]
            dataset_iter = iter(dataset)
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
                with strategy.scope():
                    D_optimizer = Adam(learning_rate=G_lrate, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
                    G_optimizer = Adam(learning_rate=D_lrate, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
            else:
                D_optimizer = Adam(learning_rate=G_lrate, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
                G_optimizer = Adam(learning_rate=D_lrate, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)

            #change model
            if prev_res != -1 and change_model:
                #increase models
                gen.save_weights('generator.h5')
                gen = networks2.generator(resolution)
                gen.load_weights('generator.h5', by_name=True)
                os.remove('generator.h5') 

                disc.save_weights('discriminator.h5')
                disc = networks2.Combined_Discriminator(resolution)
                disc.built = True
                disc.load_weights('discriminator.h5', by_name=True)
                os.remove('discriminator.h5') 
            
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

            print('Finished changing resolution')

        # Run training ops.
        for repeat in range(minibatch_repeats):
            #print(repeat)
            if resolution == 32:
                gen_loss, disc_loss, global_loss, local_loss = training_step_tf_fuction_32(dataset_iter, lod_in_value)
            elif resolution == 64:
                gen_loss, disc_loss, global_loss, local_loss = training_step_tf_fuction_64(dataset_iter, lod_in_value)
            elif resolution == 128:
                gen_loss, disc_loss, global_loss, local_loss = training_step_tf_fuction_128(dataset_iter, lod_in_value)
            elif resolution == 256:
                gen_loss, disc_loss, global_loss, local_loss = training_step_tf_fuction_256(dataset_iter, lod_in_value)
            #print(lossval.numpy())

        #Run Test


        # get stats and save model
        gen_loss_train.append(gen_loss.numpy())
        disc_loss_train.append(disc_loss.numpy())
        global_loss_train.append(global_loss.numpy()) 
        local_loss_train.append(local_loss.numpy())

        if curr_image % (global_batch_size * 100) == 0:
            print(curr_image)
            print(time.time() - train_time )
            print(lod_in_value)
            # get stats
            print(gen_loss.numpy())
            print(disc_loss.numpy())
            print(global_loss.numpy())
            print(local_loss.numpy())
            print('\n')
            train_time = time.time()
            grid = construct_grid_to_save_pgan(gw, gh, reals, grid, cvae_model=cvae, gen_model=gen, lod_in=lod_in_value)
            save_grid_pgan(gw, gh,grid, step=curr_image)

        if curr_image % (global_batch_size * 10000) == 0:
            # save model
            gen.save_weights(outpath+'saved_models/generator_'+str(int(curr_image/1000))+'.h5')
            disc.save_weights(outpath+'saved_models/discriminator_'+str(int(curr_image/1000))+'.h5')
            pickle.dump({'gen_loss': gen_loss_train, 'disc_loss': disc_loss_train, 'global_loss': global_loss_train, 'local_loss': local_loss_train}, open(outpath+'losses.pkl', 'wb'))
            print('Model Saved')
        
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
    
    use_gpus = True
    use_multi_gpus = False
 

    #training repeats
    n_epochs = 300

    #initialization of paths
    #tfrecord_dir = '/home/renato/dataset'
    #tfrecord_dir = '/home/renato/dataset/satimages'
    tfrecord_dir =  '/eos/user/r/redacost/progan/tfrecords1024/'

    start_init = time.time()
    f = [0.9, 0.1] # train, test fractions might be necessary



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


    #initialize models and optimizer
    if use_stategy:
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        num_replicas = strategy.num_replicas_in_sync
        global_batch_size = 32 * num_replicas
        print('Global batch size: ' + str(global_batch_size))
        with strategy.scope():
            cvae = networks2.CVAE(resolution=256, base_filter=32,latent_dim=512)
    else:
        cvae = networks2.CVAE(resolution=256, base_filter=32,latent_dim=128)

    opt = Adam(learning_rate=LR, epsilon=EPSILON)

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

    tfr_file = sorted(glob.glob(os.path.join(tfrecord_dir, '*r08.tfrecords')))[0]


    shuffle_mb = 10350     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
    #shuffle_mb = 27350 

    dset = tf.data.TFRecordDataset(tfr_file, compression_type='')
    # lod_dataset = 0
    # dataset = dataset_list[lod_dataset]

    #get dataset
    
    
    #use parse function
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #shuffle
    if shuffle_mb > 0:
        dset = dset.shuffle(shuffle_mb)
    #repeat    
    dset = dset.repeat()
    #prefetch
    dset = dset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #batch
    print('Global batch size: ' + str(global_batch_size))
    dset = dset.batch(global_batch_size)

    if use_stategy:
        dset = strategy.experimental_distribute_dataset(dset)
        
    dataset_iter = iter(dset)

    print('Dataset Read')

    
    dataset_size = 10350 
    batch_repeats = int(dataset_size / global_batch_size)

    print('N_batches = ' + str(batch_repeats))

    #do any necessary data processing

    # Start training
    epoch_start = time.time()

    #setup image grid
    gw, gh, reals, fakes, grid = setup_image_grid([3,128,128],  m_size = '1080p')

    def training_step(batch):
        mirror_augment = False
        drange_net = [-1,1]
        drange_data = [0, 255]

        batch = process_reals_cvae(batch, drange_data, drange_net)
        #return batch

        reconstruction_loss, kl_loss = loss.wasserstein_auto_encoder_loss(cvae, batch, global_batch_size, opt, kernel_type = 'rbf')


        return  reconstruction_loss, kl_loss

    if use_stategy:
        @tf.function
        def training_step_tf_fuction(dataset):
            reconstruction_loss, kl_loss = strategy.run(training_step, args=(next(dataset),))

            reconstruction_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, reconstruction_loss, axis=None)
            kl_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, kl_loss, axis=None)
            #rgb_angle = strategy.reduce(tf.distribute.ReduceOp.SUM, rgb_angle, axis=None)

            return reconstruction_loss, kl_loss
    else:
        @tf.function
        def training_step_tf_fuction(dataset):
            reconstruction_loss, kl_loss = training_step(next(dataset),)
            return reconstruction_loss, kl_loss
        
    train_time = time.time()      

    reconstruction_loss_train = []
    kl_loss_train = []

    for epoch in range(n_epochs):
        print('Epoch: ' + str(epoch))

        train_time = time.time() 

        # Run training ops.
        for _ in range(batch_repeats):
            reconstruction_loss, kl_loss = training_step_tf_fuction(dataset_iter)

            #Run Test

            reconstruction_loss_train.append(reconstruction_loss.numpy())
            kl_loss_train.append(kl_loss.numpy())

        
        # print(time.time() - train_time )
        # print('Reconstruction Loss ' + str(reconstruction_loss.numpy()))
        # print('Kl Loss ' + str(kl_loss.numpy()))

        # save model
        if epoch % 10 == 0:
            cvae.save_weights('saved_models/vae/cvae_'+str(int(epoch))+'.h5')
            grid = construct_grid_to_save(gw, gh, reals, fakes, grid, model=cvae, step=epoch)
            save_grid(gw, gh,grid, step=epoch)
            # print('Model Saved')
            print(time.time() - train_time )
            print('Reconstruction Loss ' + str(reconstruction_loss.numpy()))
            print('Kl Loss ' + str(kl_loss.numpy()))
            #print('rgb_angle ' + str(rgb_angle.numpy())) 
        pickle.dump({'reconstruction_loss': reconstruction_loss_train, 'kl_loss': kl_loss_train}, open('saved_models/vae/losses_'+str(lr)+'.pkl', 'wb'))
        

    cvae.save_weights('saved_models/vae/cvae_models/cvae_Final.h5')
    grid = construct_grid_to_save(gw, gh, reals, fakes, grid, model=cvae, step=epoch)
    save_grid(gw, gh,grid, step=epoch)

    total_loss = reconstruction_loss + kl_loss

    return cvae #total_loss

def get_parser():
    parser = argparse.ArgumentParser(description='Progressive GAN Params')
    parser.add_argument('--multi_node', action='store', default=False)
    parser.add_argument('--workers', nargs='+', default='') #use like this --workers 10.1.10.58:12345 10.1.10.250:12345
    parser.add_argument('--index', action='store', type=int, default=0)
    parser.add_argument('--use_gs', action='store', default=False)
    parser.add_argument('--datapath', action='store', default='') #'/eos/user/r/redacost/progan/tfrecords1024/'
    parser.add_argument('--outpath', action='store', default='')
    #
    parser.add_argument('--init_res', action='store', type=int, default=32, help='initial resolution of the progressive gan')
    parser.add_argument('--max_res', action='store', type=int, default=256, help='maximum resolution based on the dataset')
    parser.add_argument('--change_model', action='store', default=False, help='if the model is increased based on resolution, as the model is now this is always false')
    #
    parser.add_argument('--lod_training_kimg', action='store', type=int, default=300)
    parser.add_argument('--lod_transition_kimg', action='store', type=int, default=1500)
    parser.add_argument('--minibatch_base', action='store', type=int, default=32)
    #parser.add_argument('--minibatch_dict', action='store', type=int, default=64)
    #parser.add_argument('--max_minibatch_per_gpu', action='store', type=int, default=64)
    #parser.add_argument('--G_lrate_dict', action='store', type=int, default=64)
    #parser.add_argument('--D_lrate_dict', action='store', type=int, default=64)
    parser.add_argument('--total_kimg', action='store', type=int, default=1200)
    parser.add_argument('--minibatch_repeats', action='store', type=int, default=4)
    parser.add_argument('--D_repeats', action='store', type=int, default=1)
    
    #parser.add_argument('--do_profiling', action='store', default=False)
    return parser



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    #i = 0
    for gpu in gpus:
        #print(i)
        tf.config.experimental.set_memory_growth(gpu, True)
    #np.random.seed(1000)
    #tf.random.set_seed(np.random.randint(1 << 31))
    train_cycle()
#    gen = networks.named_generator_model(256)
#    gen.load_weights('generator_Final.h5', by_name=True)
#    image = get_image(1)
#    generate_image(gen, image, 'fakeimg.png', 256, lod_in=0.0)
    #model = encoder_train_cycle(lr=8.5e-5)
    #generate_decoded_image(model)
    #study = optuna.create_study()
    #study.optimize(objective, n_trials=100, callbacks=[print_best_callback])
    #gw, gh, reals, fakes, grid = setup_image_grid([3,128,128],  m_size = '1080p')
    #grid = construct_grid_to_save(gw, gh, reals, fakes, grid, model=None, step=0)
    #save_grid(gw, gh,grid)
    print('Finished Training')
