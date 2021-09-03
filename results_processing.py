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

    plt.ylim([0, 10])

    fig.savefig("saved_models/vae/reconstruction_loss.png")

    fig, ax = plt.subplots()
    ax.plot(t, kl_loss)

    print('Loss ' + str(  kl_loss[-1]))

    #plt.ylim([0, .3])

    fig.savefig("saved_models/vae/mmd_loss.png")

    # fig, ax = plt.subplots()
    # ax.plot(t, gen_loss)
    # ax.plot(t, disc_loss)
    # fig.savefig("losses.png")
    return

def get_images_grid(n_images):
    images= []
    for imgi in range(n_images):
        filename = 'cond4/'+str(imgi+1)+'.png'
        #filename = 'img.png'
        if os.path.isfile(filename):
            im=Image.open(filename)
            im.load()
            im = np.asarray(im, dtype=np.float32 )
            im=np.transpose(im, (2, 0, 1))
        images.append(im)

    if n_images > 1:
        for imgi in range(n_images-1):
            filename = 'cond4/'+str(imgi*5+6)+'.png'
            #filename = 'img.png'
            if os.path.isfile(filename):
                im=Image.open(filename)
                im.load()
                im = np.asarray(im, dtype=np.float32 )
                im=np.transpose(im, (2, 0, 1))
            images.append(im)


    return images


def setup_image_grid(dataset_shape, n_images=1, m_size= '1080p'):

    # Select size
    gw = n_images; gh = n_images
    # if m_size == '1080p':
    #     gw = np.clip(1920 // dataset_shape[2], 3, 32)
    #     gw = gw - (gw % 2)
    #     gh = np.clip(1080 // dataset_shape[1], 2, 32)
    # if m_size == '4k':
    #     gw = np.clip(3840 // dataset_shape[2], 7, 32)
    #     gw = gw - (gw % 2)
    #     gh = np.clip(2160 // dataset_shape[1], 4, 32)

    #size = dataset_shape
    #size = int(size / 2)
    size = 128 #dataset_shape

    #get images
    images = get_images_grid(n_images)

    img_counter = 0

    # Fill in reals and labels.
    #reals = np.zeros([int(gw * gh)] + dataset_shape, dtype=np.float32)
    #fakes = np.zeros([int((gw / 2) * gh)] + dataset_shape, dtype=np.float32)
    grid = np.zeros([(gw * 2) * (gh * 2)] + dataset_shape, dtype=np.float32)
    for idx in range(gw * gh):

        if (idx < n_images):
            real = images[img_counter]
            img_counter += 1

            corner1= real[:, :(size),:(size)]
            corner2= real[:, (size):,:(size)]
            corner3= real[:, :(size),(size):]
            corner4= real[:, (size):,(size):]

            corner1 = adjust_dynamic_range(corner1, [0,255], [-1,1])
            corner2 = adjust_dynamic_range(corner2, [0,255], [-1,1])
            corner3 = adjust_dynamic_range(corner3, [0,255], [-1,1])
            corner4 = adjust_dynamic_range(corner4, [0,255], [-1,1])

            grid[idx*2] = corner1
            grid[idx*2+1] = corner3
            grid[idx*2 + n_images * 2] = corner2
            grid[idx*2 + n_images * 2 +1] = corner4

        if (idx % n_images == 0) and idx != 0:
            real = images[img_counter]
            img_counter += 1

            corner1= real[:, :(size),:(size)]
            corner2= real[:, (size):,:(size)]
            corner3= real[:, :(size),(size):]
            corner4= real[:, (size):,(size):]

            corner1 = adjust_dynamic_range(corner1, [0,255], [-1,1])
            corner2 = adjust_dynamic_range(corner2, [0,255], [-1,1])
            corner3 = adjust_dynamic_range(corner3, [0,255], [-1,1])
            corner4 = adjust_dynamic_range(corner4, [0,255], [-1,1])

            grid[idx*2 *2] = corner1
            grid[idx*2 *2 +1] = corner3
            grid[idx*2 *2 + n_images * 2] = corner2
            grid[idx*2 *2 + n_images * 2 +1] = corner4



        # corner1= real[:, :(size),:(size)]
        # corner2= real[:, (size):,:(size)]
        # corner3= real[:, :(size),(size):]
        # corner4= np.zeros([3, 128, 128])

        # image_right = tf.concat([corner3, corner4], axis=1)
        # image_left = tf.concat([corner1, corner2], axis=1)
        # image_full = tf.concat([image_left, image_right], axis=2)

        #grid[idx] = image_full
        #reals[idx] = real

    # Generate latents.
    return gw, gh, grid

def generate_big_image(n_images, cvae_model, gen_model, lod_in = 0.0):
    gw, gh, grid = setup_image_grid(dataset_shape=[3,128,128], n_images=n_images, m_size= '1080p')
    total_n_images = n_images * 2 * n_images * 2
    for imgi in range(total_n_images):
        if imgi > n_images * 2 and imgi < (total_n_images - n_images * 2) and (imgi % (n_images * 2)) != 0 and imgi % (n_images * 2) != (n_images * 2 - 1):

            corner1 = np.zeros((1, 3, 128, 128))
            corner2 = np.zeros((1, 3, 128, 128))
            corner3 = np.zeros((1, 3, 128, 128))
            #corner4 = np.zeros((1, 3, 128, 128))

            corner1[0]= grid[imgi]
            corner2[0]= grid[imgi + 1]
            corner3[0]= grid[imgi + n_images * 2]
            #corner4= grid[imgi + + n_images * 2 +1]

            noise_shape = 512

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

            #grid_fakes = adjust_dynamic_range(grid_fakes, [-1,1], [0,255])

            grid[imgi + n_images * 2 +1] = grid_fakes#.transpose(1, 2, 0) # CHW -> HWC

            #image = adjust_dynamic_range(image, drange_net, drange_data)

        grid_to_save = []

    for el in grid:
        #gridi = el.transpose(1, 2, 0) 
        el = adjust_dynamic_range(el, [-1,1], [0,255])
        grid_to_save.append(el)

    return gw, gh, grid_to_save

def save_grid(gw, gh,grid,dataset_shape=None,step=0):
    num, img_w, img_h = len(grid), grid[0].shape[2], grid[0].shape[1]
    print(num, img_w, img_h)
    print(gw,gh)


    save_grid = np.zeros([grid[0].shape[0]] + [gh * 2 * img_h, gw * 2 * img_w], dtype=np.float32)
    for idx in range(num):
        x = (idx % (gw * 2)) * img_w
        y = (idx // (gw * 2)) * img_h
        print(x,y)
        save_grid[..., y : y + img_h, x : x + img_w] = grid[idx]

    image = save_grid.transpose(1, 2, 0) 
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB'
    image = Image.fromarray(image, format).save('generated_image.png')

    return save_grid


if __name__ == "__main__":
    #plot_loss()
    #cvae_plot_loss()

    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.set_visible_devices(physical_devices[1:], 'GPU')


    gen = networks2.generator(256, num_replicas = 1)
    gen.built = True
    gen.load_weights('models/generator_8000.h5')
    cvae = networks2.CVAE(resolution=256, base_filter=32,latent_dim=1024)
    cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
    cvae.load_weights('models/cvae_Final.h5')

    n_images = 2

    gw, gh, grid_to_save = generate_big_image(n_images, cvae, gen, lod_in = 0.0)
    save_grid(gw, gh, grid_to_save, dataset_shape=None,step=0)
        
    # gen = networks2.generator(256)
    # step = 640
    # gen.load_weights('./saved_models/generator_'+str(step)+'.h5', by_name=True)
    # image = get_image(1)
    # res = 256
    # res = int(np.log2(res))
    # lod_in = calculate_lod(step*1000,res)
    # print(lod_in)
    # generate_image(gen, image, 'fakeimg.png', 256, lod_in=lod_in)
