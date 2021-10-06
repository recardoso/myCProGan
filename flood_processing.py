import cv2 as cv
import numpy as np
from PIL import Image
import tensorflow as tf
import os

import networks2

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_byte_feature(val):
    # if not isinstance(val, list):
    #     print('error')
    #     val = [val]
    featurelist = np.array(val)
    flat = featurelist.flatten()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=flat))

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def snapshot(dataset, n_images=1, save=False):

    dataset_iter = iter(dataset)
    images = []

    # for el in dataset.take(n_images):
    #     images.append(el.numpy()[0])
    #     #print(repr(el))

    #n_images = 10

    for _ in range(n_images):
        dataset_el = next(dataset_iter)
        dataset_el = dataset_el['images'].numpy()[0]
        #image = (dataset.transpose(1, 2, 0).astype(np.float32)-127.5)/127.5
        #image = adjust_dynamic_range(dataset_el, [0, 255], [-1,1])
        #image = np.rint(image).clip(0, 255).astype(np.uint8)
        images.append(dataset_el)

    #print(images)


    if save:

        image = images[0]

        image = image.transpose(1, 2, 0)#.astype(np.float32)-127.5)/127.5
        #image = adjust_dynamic_range(image, [-1,1], [0, 255])
        image = np.rint(image).clip(0, 255).astype(np.uint8)

        format = 'RGB' if image.ndim == 3 else 'L'
        print(format)
        image = Image.fromarray(image, format).save('img.png')
    
    return images



def convert_to_tfrecods(dataset):

    dataset = tf.data.Dataset.from_tensor_slices(dataset)#.batch(128)

    # for f0 in dataset.take(10):
    #     print(f0)


    def serialize(feature1):
        finaldata = tf.train.Example(
            features=tf.train.Features( 
                feature={
                    'images': convert_byte_feature(feature1), #uint8
                }
            )
        )
        #seri += 1
        #print(seri)
        return finaldata.SerializeToString()

    def serialize_example(f0):
        tf_string = tf.py_function(serialize,(f0,),tf.string)
        return tf.reshape(tf_string, ())

    #for f0, f1, f2, f3 in dataset.take(1):
    #    print(serialize(f0,f1,f2,f3))

    serialized_dataset = dataset.map(serialize_example)
    #print(serialized_dataset) 
        

    #def generator():
    #    for features in dataset:
    #        yield dataset(*features)

    #serialized_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())

    #print(finaldata)

    filename ='/eos/user/r/redacost/Flood_Dataset/Flood_Images.tfrecords'
    print('Writing data in .....', filename)
    writer = tf.data.experimental.TFRecordWriter(str(filename))
    writer.write(serialized_dataset)

    return serialized_dataset

def retrieve_tfrecords(recorddatapaths, batch_size):
    recorddata = tf.data.TFRecordDataset(recorddatapaths, num_parallel_reads=tf.data.experimental.AUTOTUNE)

    #print('Start')
    #size = recorddata.cardinality().numpy()
    #print(size)

    #ds_size = sum(1 for _ in recorddata)

    
    #options = tf.data.Options()
    #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    #print(type(recorddata))

    #for rec in recorddata.take(10):
    #    print(repr(rec))

    retrieveddata = {
        'images': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True), #float32
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        data = tf.io.parse_single_example(example_proto, retrieveddata)
        data['images'] = tf.reshape(data['images'],[256,256])
        rgb_image = [data['images'],data['images'],data['images']]
        data['images'] = rgb_image
        return data

    parsed_dataset = recorddata.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(38472).repeat().batch(batch_size, drop_remainder=True)#.with_options(options)
    #print(parsed_dataset)
   
    #b = 0
    #for batch in parsed_dataset:
    #    b += 1
    #    print(b)
    #    print(batch.get('Y'))

    #for par in parsed_dataset.take(10):
    #    print(repr(par))

    #return parsed_dataset

    #print(type(parsed_dataset))

    return parsed_dataset#, ds_size

def construct_grid_to_save(gw, gh, reals, fakes, grid, model=None, step=0):
    size=128
    if model != None:
        cvae = model
    else:
        cvae = networks2.CVAE(resolution=256, base_filter=32, latent_dim=1024)
        cvae.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
        cvae.load_weights('models/cvae_Final.h5')
    for idx in range(gw * gh):
        x = idx % gw; y = idx // gw
        if idx % 2 == 0:
            continue
        if idx % 2 == 1:
            im= reals[idx // 2]
            im = adjust_dynamic_range(im, [0,255], [-1,1])
            #im= im[:,:, :(size),:(size)]
            #im= np.transpose(im, (2, 0, 1))

            batch = np.zeros((1, 3, size, size))
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

def construct_grid_to_save_pgan(gw, gh, reals, grid, cvae_model=None, gen_model=None, lod_in=0.0):
    size=128
    if cvae_model != None:
        pass
    else:
        cvae_model = networks2.CVAE(resolution=256, base_filter=32, latent_dim=1024)
        cvae_model.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
        cvae_model.load_weights('models/cvae_Final.h5')
    if gen_model != None:
        pass
    else:
        gen_model = networks2.generator()
        gen_model.built = True #subcalssed model needs to be built use tf format instead of hdf5 might solve the problem
        gen_model.load_weights('models/generator_8000.h5')
    for idx in range(gw * gh):
        x = idx % gw; y = idx // gw
        
        im= reals[idx]
        im = adjust_dynamic_range(im, [0,255], [-1,1])
        #im= im[:,:, :(size),:(size)]
        #im= np.transpose(im, (2, 0, 1))

        batch = np.zeros((1, 3, 256, 256))
        batch[0] = im
        #batch = batch[:,:, :(size),:(size)]


        corner1= batch[:,:, :(size),:(size)]
        corner2= batch[:,:, (size):,:(size)]
        corner3= batch[:,:, :(size),(size):]


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

        grid_fakes = adjust_dynamic_range(grid_fakes, [-1,1], [0,255])

        grid[idx] = grid_fakes#.transpose(1, 2, 0) # CHW -> HWC

        #image = adjust_dynamic_range(image, drange_net, drange_data)

    grid_to_save = []

    for el in grid:
        #gridi = el.transpose(1, 2, 0) 
        grid_to_save.append(el)

    return grid_to_save

def setup_image_grid(dataset, dataset_shape, m_size= '1080p', is_ae=True):

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

        images = snapshot(dataset, n_images=int((gw / 2)) * gh, save=False)

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

        images = snapshot(dataset, n_images=int(gw * gh), save=False)

        # Fill in reals and labels.
        reals = np.zeros([int(gw * gh)] + dataset_shape, dtype=np.float32)
        fakes = np.zeros([int((gw / 2) * gh)] + dataset_shape, dtype=np.float32)
        grid = np.zeros([gw * gh] + dataset_shape, dtype=np.float32)
        for idx in range(gw * gh):
            real = images[idx]

            corner1= real[:, :(size),:(size)]
            corner2= real[:, (size):,:(size)]
            corner3= real[:, :(size),(size):]
            corner4= np.zeros([3, 128, 128])

            image_right = tf.concat([corner3, corner4], axis=1)
            image_left = tf.concat([corner1, corner2], axis=1)
            image_full = tf.concat([image_left, image_right], axis=2)

            grid[idx] = image_full
            reals[idx] = real

        # Generate latents.
        return gw, gh, reals, fakes, grid

def save_grid(gw, gh,grid,dataset_shape=None,step=0,save_local='models/img.png'):
    num, img_w, img_h = len(grid), grid[0].shape[2], grid[0].shape[1]
    #print(num, img_w, img_h)


    save_grid = np.zeros([grid[0].shape[0]] + [gh * img_h, gw * img_w], dtype=np.float32)
    for idx in range(num):
        x = (idx % gw) * img_w
        y = (idx // gw) * img_h
        save_grid[..., y : y + img_h, x : x + img_w] = grid[idx]

    image = save_grid.transpose(1, 2, 0) 
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB'
    image = Image.fromarray(image, format).convert('L').save(save_local)

    return save_grid



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    image = cv.imread('/eos/user/r/redacost/Flood_Dataset/S1A_20160805T114607_ACBC_Orb_TC_compressed.tif', -1)
    ishape = image.shape
    x = ishape[0]
    y = ishape[1]
    print(x,y)
    size = 256
    nx_images = x // size #168
    ny_images = y // size #229
    print(nx_images, ny_images)
    dataset_images = []
    for x in range(0,nx_images):
        for y in range(0,ny_images):
            dataset_images.append(image[x*size:x*size+size,y*size:y*size+size])
    dataset = np.array(dataset_images)
    print(dataset.shape)
    # image=dataset[1000]
    # image = Image.fromarray(image, 'L').save('flood_image.png')

    #convert_to_tfrecods(dataset)

    

    dataset = retrieve_tfrecords('/eos/user/r/redacost/Flood_Dataset/Flood_Images.tfrecords', 16)

    print(dataset)

    # #snapshot(dataset)

    # # ae
    # # gw, gh, reals, fakes, grid = setup_image_grid(dataset, [3,128,128],  m_size = '1080p')
    # # grid = construct_grid_to_save(gw, gh, reals, fakes, grid)
    # # save_grid(gw, gh,grid,save_local='models/snapshots_vae.png')

    # #PCGAN
    gw, gh, reals, fakes, grid = setup_image_grid(dataset, [3,256,256],  m_size = '1080p', is_ae=False)
    grid = construct_grid_to_save_pgan(gw, gh, reals, grid)
    save_grid(gw, gh,grid,save_local='models/snapshots_pcgan.png')

    
