import os
import time
import numpy as np
import tensorflow as tf
import glob

#import config
#import tfutil
#import dataset
import networks
import loss
#import misc

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical, plot_model


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

    # Upscale to match the expected input/output size of the networks.
    s = tf.shape(x)
    factor = tf.cast(2 ** tf.floor(lod), tf.int32)
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    #tf.print(tf.shape(x))
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
    minibatch_base = 16
    minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
    max_minibatch_per_gpu = {256: 8, 512: 4, 1024: 2}

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

    print('dataset num of lods ' + str(len(_tf_datasets)))

    for key, value in _tf_datasets.items() :
        print(key)

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

    return lod, resolution, G_lrate, D_lrate


#----------------------------------------------------------------------------
# Training Cycle


#for 4 GPUS


def train_cycle():
    #initialization of variables
    LR=0.0015
    BETA_1 = 0.0
    BETA_2 = 0.99
    EPSILON = 1e-8

    change = True #it starts at true to be able do do any initializations
    init_res = 32 #initial resolution
    max_res = 256 #max res based on the dataset
    max_res_log2 = int(np.log2(max_res))
    num_replicas = 1
    curr_res_log2 = int(np.log2(init_res))
    prev_res = -1

    #for 4 GPUS
    lod_training_kimg = 300
    lod_transition_kimg = 1500

    minibatch_base = 16
    minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
    max_minibatch_per_gpu = {256: 8, 512: 4, 1024: 2}
    G_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}
    D_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}
    total_kimg = 12000

    minibatch_repeats = 4
    D_repeats = 1 #not used right now needs to separate discriminator and generator training 
    n_epochs = 10
    curr_image = 0

    #initialization of paths
    tfrecord_dir = '/home/renato/dataset'

    start_init = time.time()
    f = [0.9, 0.1] # train, test fractions might be necessary

    #initialize models and optimizer
    gen = networks.named_generator_model(init_res) #choose resolution
    #plot_model(gen, show_shapes=True, dpi=64)
    disc = networks.named_discriminator(init_res) #choose resolution
    #plot_model(disc, show_shapes=True, dpi=64)
    D_optimizer = Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
    G_optimizer = Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)

    #load data
    tfr_files = sorted(glob.glob(os.path.join(tfrecord_dir, '*.tfrecords')))
    dataset_list, batch_sizes_list = get_dataset(tfr_files, num_gpus=num_replicas) #all the images for all the lods

    #do any necessary data processing

    # Start training
    epoch_start = time.time()

    def training_step(batch):

        #print(batch)
        #batch_size = 120
        #batch.get_shape().as_list()[0]#.numpy()[0]
        #tf.print(batch)
        #print(batch_size)

        lod_in= tf.zeros([1, 1])
        lod_in_value = 0.0
        mirror_augment = False
        drange_net = [-1,1]
        drange_data = [0, 255]
        net_size = resolution

        batch = process_reals(batch, lod_in_value, mirror_augment, drange_data, drange_net)

        #return

        #lod_assign_ops = [tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_gpu.find_var('lod'), lod_in)] #this is probably outside and used to update the dataset/model
        #reals_gpu = process_reals(reals_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net) #need to setup this logic
        #labels_gpu = labels_split[gpu] #probabilly not necessary

        gen_loss = loss.Generator_loss(gen, disc, batch, batch_size, G_optimizer, lod_in=lod_in, training_set=None, cond_weight = 1.0, network_size=net_size)
        disc_loss = loss.Discriminator_loss(gen, disc, batch, batch_size, D_optimizer, lod_in=lod_in, training_set=None, labels=None, wgan_lambda = 10.0, wgan_epsilon = 0.001, wgan_target = 1.0, cond_weight = 1.0,  network_size=net_size)  

        #tf.print(gen_loss)

        return gen_loss

    @tf.function
    def training_step_tf_fuction_32(dataset):
        lossval = training_step(next(dataset))
        return lossval

    @tf.function
    def training_step_tf_fuction_64(dataset):
        lossval = training_step(next(dataset))
        return lossval

    @tf.function
    def training_step_tf_fuction_128(dataset):
        lossval = training_step(next(dataset))
        return lossval

    for epoch in range(n_epochs):
        print(epoch)
        # update model / variables (lr, lod, dataset) if needed 

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

        if epoch >= 5:
            resolution = 64
            lod = 2

        if prev_res != resolution:
            print('Increase Resoltion from %d to %d' % (prev_res, resolution))
            #lod is 6 for 4x4 and 0 for 256x256 (Should be change to match lod obtainned from TrainingSchedule)
            resolution_log2 = int(np.log2(resolution))
            lod_dataset = max_res_log2 - resolution_log2
            #get dataset and new batch size
            dataset = dataset_list[lod_dataset]
            dataset_iter = iter(dataset)
            batch_size = int(batch_sizes_list[lod_dataset] / num_replicas)
            

            if prev_res != -1:
                #increase models
                gen.save_weights('generator.h5')
                gen = networks.named_generator_model(resolution)
                gen.load_weights('generator.h5', by_name=True)

                disc.save_weights('discriminator.h5')
                disc = networks.named_discriminator(resolution)
                disc.load_weights('discriminator.h5', by_name=True)

            #optimizer learning rate (and reset?)

            prev_res = resolution

        # Run training ops.
        for repeat in range(minibatch_repeats):
            #print(repeat)
            if resolution == 32:
                lossval = training_step_tf_fuction_32(dataset_iter)
            elif resolution == 64:
                lossval = training_step_tf_fuction_64(dataset_iter)
            #print(lossval.numpy())

        #Run Test


        #get stats and save model
        # gen.save_weights('generator.h5')
        # gen = networks.named_generator_model(64)
        # gen.load_weights('generator.h5', by_name=True)

        # print('Loaded Model')

    return



if __name__ == "__main__":
   train_cycle()
   print('Finished Training')