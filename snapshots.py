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
import networks2
import loss
from train import *
#import misc
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical, plot_model
#import tensorflow_datasets as tfds

from PIL import Image

import pickle
#import optuna

def get_image(imgi):
    filename = 'cond4/'+str(imgi)+'.png'
    if os.path.isfile(filename):
        im=Image.open(filename)
        im.load()
        im = np.asarray(im, dtype=np.float32 )
        im=np.transpose(im, (2, 0, 1))

    return im

def generate_image(gen, img, saveloc, fullsize, lod_in=0.0):
    drange_net              = [-1,1]       # Dynamic range used when feeding image data to the networks.
    size= int(fullsize / 2)
    drange_data = [0, fullsize-1]

    grid_reals = np.zeros((1, 3, 256, 256))
    grid_reals[0] = img


    #generate latent with concatenation
    real1= grid_reals[:,:, :(size),:(size)]
    real2= grid_reals[:,:, (size):,:(size)]
    real3= grid_reals[:,:, :(size),(size):]
    real1=(real1.astype(np.float32)-127.5)/127.5
    real2=(real2.astype(np.float32)-127.5)/127.5
    real3=(real3.astype(np.float32)-127.5)/127.5
    print('real3 shape' + str(real3.shape))
    latents = np.random.randn(1, 3, 128, 128)
    left = np.concatenate((real1, real2), axis=2)
    right = np.concatenate((real3, latents), axis=2)
    lat_and_cond = np.concatenate((left, right), axis=3)


    # Conditional GAN Loss
    # real1= grid_reals[:,:, :(size),:(size)]
    # real2= grid_reals[:,:, (size):,:(size)]
    # real3= grid_reals[:,:, :(size),(size):]
    # real4= grid_reals[:,:, :(size), :(size)]
    
   
    # #generate fakes
    # latents = tf.random.normal([1, 3, size, size])
    # left = tf.concat([real1, real2], axis=2)
    # right = tf.concat([real3, latents], axis=2)
    # lat_and_cond = tf.concat([left, right], axis=3)


    #generate 128x128 image
    fake_images_out_small = gen([lat_and_cond, lod_in], training=False)
    fake_images_out_small = fake_images_out_small.numpy()


    #concatenate generated image with real images
    fake_image_out_right =np.concatenate((real3, fake_images_out_small), axis=2)
    fake_image_out_left = np.concatenate((real1, real2), axis=2)
    grid_fakes = np.concatenate((fake_image_out_left, fake_image_out_right), axis=3)

    #grid_fakes = grid_fakes.numpy()

    #print(grid_fakes)
    grid_fakes = grid_fakes[0]


    image = grid_fakes.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange_net, drange_data)
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'
    image = Image.fromarray(image, format).save(saveloc)

    return

def snapshot(datapath, resolution_log2=256, n_images=1, save=False):
    #tfrecord_dir = '/home/renato/dataset/satimages'
    #tfrecord_dir =  '/eos/user/r/redacost/progan/tfrecords1024/'
    tfrecord_dir = datapath
    tfr_files = sorted(glob.glob(os.path.join(tfrecord_dir, '*.tfrecords')))
    minibatch_base = 32
    minibatch_dict = {4: 1024, 8: 512, 16: 256, 32: 64, 64: 64, 128: 32}
    max_minibatch_per_gpu = {256: 16, 512: 8, 1024: 8}
    dataset_list, batch_sizes_list = get_dataset(tfr_files, flood=True, num_gpus=1, minibatch_base = minibatch_base, minibatch_dict = minibatch_dict, max_minibatch_per_gpu = max_minibatch_per_gpu)
    lod_dataset = int(np.log2(256)) - int(np.log2(resolution_log2))
    dataset = dataset_list[lod_dataset]
    print(dataset)
    print(lod_dataset)
    #(train, val, test) = tfds.load("eurosat/rgb", split=["train[:100%]", "train[80%:90%]", "train[90%:]"])

    # def prepare_training_data(datapoint):
    #     input_image = datapoint["image"]
    #     data = tf.transpose(input_image, [2,0,1])
    #     return data

    # dataset = train.map(prepare_training_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    dataset_iter = iter(dataset)
    images = []

    # for el in dataset.take(n_images):
    #     images.append(el.numpy()[0])
    #     #print(repr(el))


    for _ in range(n_images):
        dataset_el = tf.cast(next(dataset_iter).numpy()[0], tf.float32) 
        #image = (dataset.transpose(1, 2, 0).astype(np.float32)-127.5)/127.5
        #image = adjust_dynamic_range(dataset_el, [0, 255], [-1,1])
        #image = np.rint(image).clip(0, 255).astype(np.uint8)
        images.append(dataset_el)

    #print(dataset)

    #image = (dataset.transpose(1, 2, 0).astype(np.float32)-127.5)/127.5

    #image = adjust_dynamic_range(image, [-1,1], [0, 255])
    #image = np.rint(image).clip(0, 255).astype(np.uint8)
    if save:
        format = 'RGB' #if image.ndim == 3 else 'L'
        image = Image.fromarray(dataset, format).save('img.png')
    
    return images

def generate_decoded_image(model = None):
    drange_net  = [-1,1]       # Dynamic range used when feeding image data to the networks.
    drange_data = [0, 256-1]

    a = np.zeros((1, 3, 256, 256))
    size = 128
    a4= a[:,:, :(size), :(size)]
    imgi = 1

    filename = 'cond4/'+str(imgi)+'.png'
    #filename = 'img.png'
    if os.path.isfile(filename):
        im=Image.open(filename)
        im.load()
        im = np.asarray(im, dtype=np.float32 )
        im= np.transpose(im, (2, 0, 1))

    batch = np.zeros((1, 3, 256, 256))
    batch[0] = im
    

    batch1= batch[:,:, :(size),:(size)]
    batch2= batch[:,:, (size):,:(size)]
    batch3= batch[:,:, :(size),(size):]
    batch4= batch[:,:, :(size), :(size)]

    # batch1=(batch1.astype(np.float32))/255
    # batch2=(batch2.astype(np.float32))/255
    # batch3=(batch3.astype(np.float32))/255

    #batchleft = tf.concat([batch1, batch2], axis=3)
    #batchall3 = tf.concat([batchleft, batch3], axis=3)

    if model != None:
        cvae = model
    else:
        cvae = networks2.CVAE(resolution=256, base_filter=8, latent_dim=256)
        cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
        cvae.load_weights('saved_models/vae/cvae_Final.h5')
    latent = cvae.encode(batch1)
    #z = cvae.reparameterize(mean, logvar)
    predictions = cvae.decode(latent)
    predictions = predictions.numpy()

    #predictionsleft = predictions[:,:, :(size), :(size)]
    #predictionsmiddle = predictions[:,:, :(size), (size):2*(size)]
    #predictionsright = predictions[:,:, :(size), 2*(size):3*(size)]

    #fake_image_out_right =np.concatenate((predictionsright, a4), axis=2)
    #fake_image_out_left = np.concatenate((predictionsleft, predictionsmiddle), axis=2)
    #grid_fakes = np.concatenate((fake_image_out_left, fake_image_out_right), axis=3)

    grid_fakes = predictions[0]

    image = grid_fakes.transpose(1, 2, 0) # CHW -> HWC

    #image = adjust_dynamic_range(image, drange_net, drange_data)
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'
    image = Image.fromarray(image, format).save('saved_models/vae/decodedimg.png')

    return
    
#optuna optimization    
def objective(trial):
    lr = trial.suggest_float("learning_rate_init", 1e-5, 1e-3, log=True)

    loss = encoder_train_cycle(lr)

    return loss

def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

def setup_image_grid(datapath,dataset_shape, m_size= '1080p', is_ae=True):

    # Select size
    gw = 1; gh = 1
    if m_size == '1080p':
        gw = np.clip(1920 // dataset_shape[2], 3, 32)
        gw = gw - (gw % 2)
        gh = np.clip(1080 // dataset_shape[1], 2, 32)
    if m_size == '4k':
        gw = np.clip(3840 // dataset_shape[2], 7, 32)
        gw = gw - (gw % 2)
        gh = np.clip(2160 // dataset_shape[1], 4, 32)

    size = dataset_shape[2]

    if is_ae:

        images = snapshot(datapath,resolution_log2=size*2,n_images=int((gw / 2)) * gh, save=False)

        # Fill in reals and labels.
        reals = np.zeros([int((gw / 2) * gh)] + dataset_shape, dtype=np.float32)
        fakes = np.zeros([int((gw / 2) * gh)] + dataset_shape, dtype=np.float32)
        grid = np.zeros([gw * gh] + dataset_shape, dtype=np.float32)
        for idx in range(gw * gh):
            x = idx % gw; y = idx // gw
            if idx % 2 == 0:
                real = images[idx//2]
                grid[idx] = real[:, :(size),:(size)]
                reals[int(idx // 2)] = real[:, :(size),:(size)]
            if idx % 2 == 1:
                grid[idx] = fakes[0]

        # Generate latents.
        return gw, gh, reals, fakes, grid

    else:
        size = int(size / 2)

        images = snapshot(datapath,resolution_log2=size*2,n_images=int(gw * gh), save=False)

        # Fill in reals and labels.
        reals = np.zeros([int(gw * gh)] + dataset_shape, dtype=np.float32)
        fakes = np.zeros([int((gw / 2) * gh)] + dataset_shape, dtype=np.float32)
        grid = np.zeros([gw * gh] + dataset_shape, dtype=np.float32)
        for idx in range(gw * gh):
            real = images[idx]

            corner1= real[:, :(size),:(size)]
            corner2= real[:, (size):,:(size)]
            corner3= real[:, :(size),(size):]
            corner4= np.zeros([1, size, size])

            image_right = tf.concat([corner3, corner4], axis=1)
            image_left = tf.concat([corner1, corner2], axis=1)
            image_full = tf.concat([image_left, image_right], axis=2)

            grid[idx] = image_full
            reals[idx] = real

        # Generate latents.
        return gw, gh, reals, fakes, grid

def construct_grid_to_save(gw, gh, reals, fakes, grid, model=None, step=0,size=128):
    if model != None:
        cvae = model
    else:
        cvae = networks2.CVAE(resolution=256, base_filter=8, latent_dim=256)
        cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
        cvae.load_weights('saved_models/vae/cvae_Final.h5')
    for idx in range(gw * gh):
        x = idx % gw; y = idx // gw
        if idx % 2 == 0:
            continue
        if idx % 2 == 1:
            im= reals[idx // 2]
            im = adjust_dynamic_range(im, [0,255], [-1,1])
            #im= im[:,:, :(size),:(size)]
            #im= np.transpose(im, (2, 0, 1))

            batch = np.zeros((1, 1, size, size))
            batch[0] = im
            #batch = batch[:,:, :(size),:(size)]

            latent = cvae.encode(batch)
            #z = cvae.reparameterize(mean, logvar)
            predictions = cvae.decode(latent)
            predictions = predictions.numpy()

            grid_fakes = predictions[0]

            grid_fakes = adjust_dynamic_range(grid_fakes, [-1,1], [0,255])

            grid[idx] = grid_fakes#.transpose(1, 2, 0) # CHW -> HWC

        #image = adjust_dynamic_range(image, drange_net, drange_data)

    grid_to_save = []

    for el in grid:
        #gridi = el.transpose(1, 2, 0) 
        grid_to_save.append(el)

    return grid_to_save

def construct_grid_to_save_pgan(gw, gh, reals, grid, cvae_model=None, gen_model=None, lod_in=0.0, size=256):
    size = int(size/2)
    for idx in range(gw * gh):
        x = idx % gw; y = idx // gw
        
        im= reals[idx]
        im = adjust_dynamic_range(im, [0,255], [-1,1])
        #im= im[:,:, :(size),:(size)]
        #im= np.transpose(im, (2, 0, 1))

        batch = np.zeros((1, 1, size*2, size*2))
        batch[0] = im
        #batch = batch[:,:, :(size),:(size)]


        corner1= batch[:,:, :(size),:(size)]
        corner2= batch[:,:, (size):,:(size)]
        corner3= batch[:,:, :(size),(size):]


        noise_shape = 2048

         #generate noise image
        latents = tf.random.normal([1, noise_shape])

        ae_latent1 = cvae_model.encode(corner1)
        ae_latent2 = cvae_model.encode(corner2)
        ae_latent3 = cvae_model.encode(corner3)

        #check axis
        ae_latent_left = tf.concat([ae_latent1, ae_latent2], axis=1)
        ae_latents = tf.concat([ae_latent_left, ae_latent3], axis=1)

        corner4 = tf.cast(gen_model([latents, ae_latents, lod_in], training=False), tf.float32) 

        image_right = tf.cast(tf.concat([corner3, corner4], axis=2), tf.float32)
        image_left = tf.cast(tf.concat([corner1, corner2], axis=2), tf.float32)
        image_full = tf.concat([image_left, image_right], axis=3)

        predictions = image_full.numpy()

        grid_fakes = predictions[0]

        grid_fakes = adjust_dynamic_range(grid_fakes, [-1,1], [0,255])

        grid[idx] = grid_fakes#.transpose(1, 2, 0) # CHW -> HWC

        #image = adjust_dynamic_range(image, drange_net, drange_data)

    grid_to_save = []

    for el in grid:
        #gridi = el.transpose(1, 2, 0) 
        grid_to_save.append(el)

    return grid_to_save

def save_grid(gw, gh,grid,dataset_shape=None,step=0):
    num, img_w, img_h = len(grid), grid[0].shape[2], grid[0].shape[1]
    #print(num, img_w, img_h)


    save_grid = np.zeros([grid[0].shape[0]] + [gh * img_h, gw * img_w], dtype=np.float32)
    for idx in range(num):
        x = (idx % gw) * img_w
        y = (idx // gw) * img_h
        save_grid[..., y : y + img_h, x : x + img_w] = grid[idx]

    #image = save_grid.transpose(1, 2, 0) 
    image = np.squeeze(save_grid, axis=0)
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'L'
    image = Image.fromarray(image, format).save('saved_models/vae/snapshots/grid_'+str(step)+'.png')

    return save_grid

def save_grid_pgan(gw, gh,grid,dataset_shape=None,step=0,outpath=''):
    num, img_w, img_h = len(grid), grid[0].shape[2], grid[0].shape[1]
    #print(num, img_w, img_h)


    save_grid = np.zeros([grid[0].shape[0]] + [gh * img_h, gw * img_w], dtype=np.float32)
    for idx in range(num):
        x = (idx % gw) * img_w
        y = (idx // gw) * img_h
        save_grid[..., y : y + img_h, x : x + img_w] = grid[idx]

#     image = save_grid.transpose(1, 2, 0) 
#     image = np.rint(image).clip(0, 255).astype(np.uint8)
#     format = 'RGB'
#     image = Image.fromarray(image, format).save(outpath+'saved_models/pgan/snapshots/grid_'+str(step)+'.png')

    image = np.squeeze(save_grid, axis=0)
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'L'
    image = Image.fromarray(image, format).save(outpath+'saved_models/pgan/snapshots/grid_'+str(step)+'.png')

    return save_grid

def get_edges(datapath):
    tfrecord_dir = datapath
    tfr_files = sorted(glob.glob(os.path.join(tfrecord_dir, '*.tfrecords')))
    minibatch_base = 32
    minibatch_dict = {4: 1024, 8: 512, 16: 256, 32: 64, 64: 64, 128: 32}
    max_minibatch_per_gpu = {256: 16, 512: 8, 1024: 8}
    dataset_list, batch_sizes_list = get_dataset(tfr_files, num_gpus=1, minibatch_base = minibatch_base, minibatch_dict = minibatch_dict, max_minibatch_per_gpu = max_minibatch_per_gpu)
    lod_dataset = 0
    dataset = dataset_list[lod_dataset]
    print(dataset)

    dataset_iter = iter(dataset)
    min = 255 #0
    max = 0 #242

    l = [0] * 256

    n_images = 1000 


    for i in range(n_images):
        dataset_el = next(dataset_iter).numpy()[0]
        dataset_el = dataset_el.flatten()
        for el in dataset_el:
            l[el] = l[el] + 1
        #r = dataset_el[0]
        # d_min = np.amin(dataset_el)
        # if d_min < min:
        #     min = d_min
        #     print("min: " + str(min))
        # d_max = np.amax(dataset_el)
        # if d_max > max:
        #     max = d_max
        #     print("max: " + str(max))
        #g = dataset_el[1]
        #b = dataset_el[2]

        if i % 100 == 0:
            print(i)
            #print(dataset_el)
            #print(dataset_el.size)

    data = np.array(l)
    print(l)

    bins = np.linspace(0, 255,256) # fixed number of bins
    print(bins)

    fig, ax = plt.subplots()

    plt.xlim([0, 256+5])
    #plt.ylim([0, 10000000])

    plt.bar(bins, data)
    plt.title('Random Gaussian data (fixed number of bins)')
    plt.xlabel('variable X (20 evenly spaced bins)')
    plt.ylabel('count')

    #ax.plot()

    fig.savefig("color_hystogram.png")

if __name__ == "__main__":
    get_edges('/eos/user/r/redacost/progan/tfrecords1024/')
        