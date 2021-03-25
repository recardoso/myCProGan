import os
import time
import numpy as np
import tensorflow as tf

#import config
#import tfutil
#import dataset
import networks
#import misc


def Train_steps(dataset):
    # Get a single batch    
    image_batch = dataset.get('X')#.numpy()
    energy_batch = dataset.get('Y')#.numpy()
    ecal_batch = dataset.get('ecal')#.numpy()
    ang_batch = dataset.get('ang')#.numpy()
    #add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)
 
    # Generate Fake events with same energy and angle as data batch
    noise = tf.random.normal((batch_size_per_replica, latent_size-2), 0, 1)
    generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1)
    generated_images = generator(generator_ip, training=False)

    # Train discriminator first on real batch 
    fake_batch = BitFlip(np.ones(batch_size_per_replica).astype(np.float32))
    fake_batch = [[el] for el in fake_batch]
    labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

    with tf.GradientTape() as tape:
        predictions = discriminator(image_batch, training=True)
        real_batch_loss = compute_global_loss(labels, predictions, batch_size, loss_weights=loss_weights)
    
    gradients = tape.gradient(real_batch_loss, discriminator.trainable_variables) # model.trainable_variables or  model.trainable_weights
    
    #------------Minimize------------
    #aggregate_grads_outside_optimizer = (optimizer._HAS_AGGREGATE_GRAD and not isinstance(strategy.extended, parameter_server_strategy.))
    #gradients = optimizer_discriminator._clip_gradients(gradients)

    #--------------------------------
    
    optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) # model.trainable_variables or  model.trainable_weights

    #Train discriminato on the fake batch
    fake_batch = BitFlip(np.zeros(batch_size_per_replica).astype(np.float32))
    fake_batch = [[el] for el in fake_batch]
    labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

    with tf.GradientTape() as tape:
        predictions = discriminator(generated_images, training=True)
        fake_batch_loss = compute_global_loss(labels, predictions, batch_size, loss_weights=loss_weights)
    gradients = tape.gradient(fake_batch_loss, discriminator.trainable_variables) # model.trainable_variables or  model.trainable_weights
    #gradients = optimizer_discriminator._clip_gradients(gradients)
    optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) # model.trainable_variables or  model.trainable_weights



    trick = np.ones(batch_size_per_replica).astype(np.float32)
    fake_batch = [[el] for el in trick]
    labels = [fake_batch, tf.reshape(energy_batch, (-1,1)), ang_batch, ecal_batch]

    gen_losses = []
    # Train generator twice using combined model
    for _ in range(2):
        noise = tf.random.normal((batch_size_per_replica, latent_size-2), 0, 1)
        generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1) # sampled angle same as g4 theta   

        with tf.GradientTape() as tape:
            generated_images = generator(generator_ip ,training= True)
            predictions = discriminator(generated_images , training=True)
            loss = compute_global_loss(labels, predictions, batch_size, loss_weights=loss_weights)

        gradients = tape.gradient(loss, generator.trainable_variables) # model.trainable_variables or  model.trainable_weights
        #gradients = optimizer_generator._clip_gradients(gradients)
        optimizer_generator.apply_gradients(zip(gradients, generator.trainable_variables)) # model.trainable_variables or  model.trainable_weights

        for el in loss:
            gen_losses.append(el)

    return real_batch_loss[0], real_batch_loss[1], real_batch_loss[2], real_batch_loss[3], fake_batch_loss[0], fake_batch_loss[1], fake_batch_loss[2], fake_batch_loss[3], \
            gen_losses[0], gen_losses[1], gen_losses[2], gen_losses[3], gen_losses[4], gen_losses[5], gen_losses[6], gen_losses[7]  

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // config.num_gpus
        reals, labels   = training_set.get_minibatch_tf()
        reals_split     = tf.split(reals, config.num_gpus)
        labels_split    = tf.split(labels, config.num_gpus)
    G_opt = tfutil.Optimizer(name='TrainG', learning_rate=lrate_in, **config.G_opt)
    D_opt = tfutil.Optimizer(name='TrainD', learning_rate=lrate_in, **config.D_opt)

    for gpu in range(config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')

            lod_assign_ops = [tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_gpu.find_var('lod'), lod_in)]
            reals_gpu = process_reals(reals_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net)
            labels_gpu = labels_split[gpu]
            with tf.name_scope('G_loss'), tf.control_dependencies(lod_assign_ops):
                G_loss = tfutil.call_func_by_name(G=G_gpu, D=D_gpu,  opt=G_opt, training_set=training_set, minibatch_size=minibatch_split, reals=reals_gpu, **config.G_loss)
            with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
                D_loss = tfutil.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_split, reals=reals_gpu, labels=labels_gpu, **config.D_loss)
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()

if __name__ == "__main__":
    print('Constructing networks...')
    #latents = np.random.randn(120, 3, 128, 128)
    #networks.G_paper(latents)
    #latents = tf.random_normal([16, 3, 32, 32])
    #networks.G_paper(latents)

    real1= tf.random.normal([120, 3, 128, 128])
    real2= tf.random.normal([120, 3, 128, 128])
    real3= tf.random.normal([120, 3, 128, 128])
    #real1=(real1.astype(np.float32)-127.5)/127.5
    #real2=(real2.astype(np.float32)-127.5)/127.5
    #real3=(real3.astype(np.float32)-127.5)/127.5
    print('real3 shape' + str(real3.shape))

    latents = tf.random.normal([120, 3, 128, 128])
    left = tf.concat([real1, real2], axis=2)
    right = tf.concat([real3, latents], axis=2)
    lat_and_cond = tf.concat([left, right], axis=3)



    #fake_images_out_small = Gs.run(lat_and_cond, grid_labels, minibatch_size=120)



    latent_size = 256
    res = 256 
    #generator=networks.Generator_tf2(latent_size, res, latents_in=lat_and_cond)
    networks.Generator_model()
    #left = tf.concat([real1, real2], axis=2)
    #right = tf.concat([real3, latents], axis=2)
    #lat_and_cond = tf.concat([left, right], axis=3)
    #G = networks.Network('G', num_channels=3, resolution=256, func='networks.G_paper')
    #D = tfutil.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1],  **config.D)