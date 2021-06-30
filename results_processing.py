import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

import networks2
import tensorflow as tf
from PIL import Image

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def get_image(imgi):
    filename = 'cond4/'+str(imgi)+'.png'
    filename = 'img.png'
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

def plot_loss():
    with open('losses.pkl', 'rb') as f:
        losses = pickle.load(f)

    gen_loss = losses['gen_loss']
    disc_loss = losses['disc_loss']

    loss_entries = len(gen_loss)
    batch_size = 64 # there is a point where it switches
    final_loss_point = loss_entries * batch_size

    t = np.arange(0.0, final_loss_point, batch_size)

    fig, ax = plt.subplots()
    ax.plot(t, gen_loss)

    fig.savefig("gen_losses.png")

    fig, ax = plt.subplots()
    ax.plot(t, disc_loss)

    fig.savefig("disc_losses.png")

    fig, ax = plt.subplots()
    ax.plot(t, gen_loss)
    ax.plot(t, disc_loss)
    fig.savefig("losses.png")

def snapshot(dataset_list):
    lod_dataset = 0
    dataset = dataset_list[lod_dataset]

    dataset = next(iter(dataset)).numpy()[0]

    print(dataset)

    image = (dataset.transpose(1, 2, 0).astype(np.float32)-127.5)/127.5

    image = adjust_dynamic_range(image, [-1,1], [0, 255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' #if image.ndim == 3 else 'L'
    image = Image.fromarray(image, format).save('img.png')
    
    return dataset

def calculate_lod(cur_nimg,dataset_res_log2,lod_initial_resolution=32,lod_training_kimg=300,lod_transition_kimg=1500):
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

    lod = np.float32(lod)

    lod = tf.cast(lod, tf.float32)

    return lod 

def cvae_plot_loss():
    with open('saved_models/vae/losses_8.5e-05.pkl', 'rb') as f:
        losses = pickle.load(f)

    reconstruction_loss = losses['reconstruction_loss']
    kl_loss = losses['kl_loss']

    loss_entries = len(reconstruction_loss)
    #batch_size = 64 # there is a point where it switches
    #final_loss_point = loss_entries * batch_size

    t = np.arange(0.0, loss_entries)

    fig, ax = plt.subplots()
    ax.plot(t, reconstruction_loss)

    fig.savefig("saved_models/vae/reconstruction_loss.png")

    fig, ax = plt.subplots()
    ax.plot(t, kl_loss)

    print('Loss ' + str(  kl_loss[-1]))

    #plt.ylim([0, .3])

    fig.savefig("saved_models/vae/kl_loss.png")

    # fig, ax = plt.subplots()
    # ax.plot(t, gen_loss)
    # ax.plot(t, disc_loss)
    # fig.savefig("losses.png")
    return


if __name__ == "__main__":
    #plot_loss()
    cvae_plot_loss()
        
    # gen = networks2.generator(256)
    # step = 640
    # gen.load_weights('./saved_models/generator_'+str(step)+'.h5', by_name=True)
    # image = get_image(1)
    # res = 256
    # res = int(np.log2(res))
    # lod_in = calculate_lod(step*1000,res)
    # print(lod_in)
    # generate_image(gen, image, 'fakeimg.png', 256, lod_in=lod_in)