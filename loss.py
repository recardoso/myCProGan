# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf
import networks2
import math
#import config

#import tfutil

import pickle

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def lerp(a, b, t):
    return a + (b - a) * t

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, opt, training_set, minibatch_size, reals,
    cond_weight = 1.0): # Weight of the conditioning term.
    print('Mini-batch size G' + str(minibatch_size))
    size= int(128)    
    # Conditional GAN Loss
    real1= reals[:,:, :(size),:(size)]
    real2= reals[:,:, (size):,:(size)]
    real3= reals[:,:, :(size),(size):]
    real4= reals[:,:, :(size), :(size)]
    
   
 
    latents = tf.random_normal([minibatch_size, 3, size, size])
    left = tf.concat([real1, real2], axis=2)
    right = tf.concat([real3, latents], axis=2)
    lat_and_cond = tf.concat([left, right], axis=3)

    
    print('lat_and_cond : ' + str(lat_and_cond))
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out_small = G.get_output_for(lat_and_cond, labels, is_training=True)
    fake_image_out_right = tf.concat([real3, fake_images_out_small], axis=2)
    fake_image_out_left = tf.concat([real1, real2], axis=2)
    fake_images_out = tf.concat([fake_image_out_left, fake_image_out_right], axis=3)
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out

    

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
        loss += label_penalty_fakes * cond_weight
        

    
    
    return loss

def Generator_loss(G, combined_D, encoder, reals, minibatch_size, opt=None, lod_in=0.0, training_set=None, cond_weight = 1.0, network_size=256, global_batch_size = 1): # Weight of the conditioning term.
    print('Mini-batch size G ' + str(minibatch_size))
    size= int(network_size / 2)  
    n_gpus = int(global_batch_size/minibatch_size)  
    # Conditional GAN Loss
    real1= reals[:,:, :(size),:(size)]
    real2= reals[:,:, (size):,:(size)]
    real3= reals[:,:, :(size),(size):]
    real4= reals[:,:, :(size), :(size)]
    
    noise_shape = 2048

    corners = [real1,real2,real3]
   
    #generate fakes
    # latents = tf.random.normal([minibatch_size, 3, size, size])
    # left = tf.concat([real1, real2], axis=2)
    # right = tf.concat([real3, latents], axis=2)
    # lat_and_cond = tf.concat([left, right], axis=3)

    #check axis
    #realsleft = tf.concat([real1, real2], axis=3)
    #reals_in_row = tf.concat([realsleft, real3], axis=3)

    #ae_latents = encoder.encode(reals_in_row)
    
    #generate noise image
    latents = tf.random.normal([minibatch_size, noise_shape])
    
    ae_latent1 = encoder.encode(real1)
    ae_latent2 = encoder.encode(real2)
    ae_latent3 = encoder.encode(real3)

    #check axis
    ae_latent_left = tf.concat([ae_latent1, ae_latent2], axis=1)
    ae_latents = tf.concat([ae_latent_left, ae_latent3], axis=1)
    
    #tf.print(tf.shape(ae_latents))



    
    print('lat_and_cond : ' + str(latents))

    with tf.GradientTape() as tape:
        fake_images_out_small = G([latents, ae_latents, lod_in], training=True)
        fake_image_out_right = tf.concat([real3, fake_images_out_small], axis=2)
        fake_image_out_left = tf.concat([real1, real2], axis=2)
        fake_images_out = tf.concat([fake_image_out_left, fake_image_out_right], axis=3)

        #global Discriminator
        global_fake_scores_out = fp32(combined_D.use_global_discriminator(fake_images_out, lod_in))

        #local Discriminator
        local_fake_scores_out = fp32(combined_D.use_local_discriminator(fake_images_out_small, lod_in))

        join_scores = (global_fake_scores_out + local_fake_scores_out) / 2
        #tf.print(fake_scores_out)
        loss = -join_scores
        g_loss = tf.reduce_mean(loss) / n_gpus


        
    gradients =  tape.gradient(g_loss, G.trainable_variables) # model.trainable_variables or  model.trainable_weights
    
    opt.apply_gradients(zip(gradients, G.trainable_variables)) # model.trainable_variables or  model.trainable_weights
    
    
    return g_loss

def Generator_test_loss(G, combined_D, encoder, reals, minibatch_size, opt=None, lod_in=0.0, training_set=None, cond_weight = 1.0, network_size=256, global_batch_size = 1): # Weight of the conditioning term.
    print('Mini-batch size G ' + str(minibatch_size))
    size= int(network_size / 2)  
    n_gpus = int(global_batch_size/minibatch_size)  
    # Conditional GAN Loss
    real1= reals[:,:, :(size),:(size)]
    real2= reals[:,:, (size):,:(size)]
    real3= reals[:,:, :(size),(size):]
    real4= reals[:,:, :(size), :(size)]
    
    noise_shape = 2048

    corners = [real1,real2,real3]
    
    #generate noise image
    latents = tf.random.normal([minibatch_size, noise_shape])
    
    ae_latent1 = encoder.encode(real1, training_flag=False)
    ae_latent2 = encoder.encode(real2, training_flag=False)
    ae_latent3 = encoder.encode(real3, training_flag=False)

    #check axis
    ae_latent_left = tf.concat([ae_latent1, ae_latent2], axis=1)
    ae_latents = tf.concat([ae_latent_left, ae_latent3], axis=1)
    
    #tf.print(tf.shape(ae_latents))
    
    print('lat_and_cond : ' + str(latents))

    #with tf.GradientTape() as tape:
    fake_images_out_small = G([latents, ae_latents, lod_in], training=False)
    fake_image_out_right = tf.concat([real3, fake_images_out_small], axis=2)
    fake_image_out_left = tf.concat([real1, real2], axis=2)
    fake_images_out = tf.concat([fake_image_out_left, fake_image_out_right], axis=3)

    #global Discriminator
    global_fake_scores_out = fp32(combined_D.use_global_discriminator(fake_images_out, lod_in, training_flag=False))

    #local Discriminator
    local_fake_scores_out = fp32(combined_D.use_local_discriminator(fake_images_out_small, lod_in, training_flag=False))

    join_scores = (global_fake_scores_out + local_fake_scores_out) / 2
    #tf.print(fake_scores_out)
    loss = -join_scores
    g_loss = tf.reduce_mean(loss) / n_gpus
    
    return g_loss

def original_Generator_loss(G, D, reals, minibatch_size, opt=None, lod_in=0.0, training_set=None, cond_weight = 1.0, network_size=256, global_batch_size = 1): # Weight of the conditioning term.
    print('Mini-batch size G ' + str(minibatch_size))
    size= int(network_size)    
    n_gpus = int(global_batch_size/minibatch_size)
 
    #generate fakes
    latents = tf.random.normal([minibatch_size, 3, size, size])  
    print('lat_and_cond : ' + str(latents))
    #tf.print(latents)

    with tf.GradientTape() as tape:
        fake_images_out = G([latents, lod_in], training=True)
        fake_scores_out, fake_labels_out = D([fake_images_out, lod_in], training=True) #convert to f32
        #tf.print(tf.shape(fake_images_out))
        loss = -fake_scores_out 


        g_loss = tf.reduce_mean(loss) / n_gpus
        #tf.print(loss)
        #loss = tf.nn.compute_average_loss(loss)#, global_batch_size=global_batch_size)
        #tf.print(loss)

    
    #add label penalty (are there any labels???)
    #if D.output_shapes[1][1] > 0:
    #    with tf.name_scope('LabelPenalty'):
    #        label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
    #    loss += label_penalty_fakes * cond_weight
        
    gradients =  tape.gradient(g_loss, G.trainable_variables) # model.trainable_variables or  model.trainable_weights
    opt.apply_gradients(zip(gradients, G.trainable_variables)) # model.trainable_variables or  model.trainable_weights
    
    
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D,opt, training_set, minibatch_size, reals, labels,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.
    print('Mini-batch size D' + str(minibatch_size))
    size= int(128)
    print('real shape' + str(reals.shape))
    
    real1= reals[:,:, :(size),:(size)]
    real2= reals[:,:, (size):,:(size)]
    real3= reals[:,:, :(size),(size):]
    real4= reals[:,:, :(size), :(size)]
    
   
 
    latents = tf.random_normal([minibatch_size, 3, size, size])
    left = tf.concat([real1, real2], axis=2)
    right = tf.concat([real3, latents], axis=2)
    lat_and_cond = tf.concat([left, right], axis=3)
    
    
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out_small = G.get_output_for(lat_and_cond, labels, is_training=True)
    fake_image_out_right = tf.concat([real3, fake_images_out_small], axis=2)
    fake_image_out_left = tf.concat([real1, real2], axis=2)
    fake_images_out = tf.concat([fake_image_out_left, fake_image_out_right], axis=3)
    
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

#----------------------------------------------------------------------------

def Discriminator_loss(G, D, reals, minibatch_size, opt, lod_in=0.0, training_set=None, labels=None,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0,      # Weight of the conditioning terms.
    global_batch_size = 1,
    network_size=256):

    #loss scalling (it is not doing loss scalling) !!!!
    n_gpus = int(global_batch_size/minibatch_size)
    print('Mini-batch size D ' + str(minibatch_size))
    size= int(network_size / 2) 
    print('real shape' + str(reals.shape))
    
    #get reals:
    real1= reals[:,:, :(size),:(size)]
    real2= reals[:,:, (size):,:(size)]
    real3= reals[:,:, :(size),(size):]
    real4= reals[:,:, :(size), :(size)]
    
    #generate noise image
    latents = tf.random.normal([minibatch_size, 3, size, size])
    left = tf.concat([real1, real2], axis=2)
    right = tf.concat([real3, latents], axis=2)
    lat_and_cond = tf.concat([left, right], axis=3)
    
    #get labels???
    #labels = training_set.get_random_labels_tf(minibatch_size)

    #needs two gradient tapes because we use the gradeients from the mixed images for the penalties
    with tf.GradientTape() as grad_total_tape:
        #first pass through the discriminator with fake and real data

        #generate fake image
        fake_images_out_small = G([lat_and_cond, lod_in], training=True)
        #tf.print(fake_images_out_small)
        #fake_images_out_small = G.get_output_for(lat_and_cond, labels, is_training=True)
        fake_image_out_right = tf.concat([real3, fake_images_out_small], axis=2)
        fake_image_out_left = tf.concat([real1, real2], axis=2)
        fake_images_out = tf.concat([fake_image_out_left, fake_image_out_right], axis=3)

        real_scores_out = fp32(D([reals, lod_in], training=True))
        fake_scores_out = fp32(D([fake_images_out, lod_in], training=True))
        #real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
        #fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
        loss = fake_scores_out - real_scores_out

        #calculate gradient penalty inside a new gradient tape
        #mixing factors
        mixing_factors = tf.random.uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        #mixed images
        mixed_images_out = lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)

        with tf.GradientTape() as grad_penalty_tape:
            grad_penalty_tape.watch(mixed_images_out)
            #get mixed scores
            mixed_scores_out = fp32(D([mixed_images_out, lod_in], training=True))

            #final mixed loss
            mixed_loss = tf.math.reduce_sum(mixed_scores_out) #might need to change?

            #tf.print(mixed_loss)

        #get mixed gradients
        mixed_grads = fp32(grad_penalty_tape.gradient(mixed_loss, [mixed_images_out])[0])

        # normalize gradients
        mixed_norms = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(mixed_grads), axis=[1,2,3]))

        #apply penalty
        gradient_penalty = tf.math.square(mixed_norms - wgan_target)

        #tf.print(mixed_loss)

        loss += gradient_penalty * (wgan_lambda / (wgan_target**2))


        #epsilon penalty
        epsilon_penalty = tf.math.square(real_scores_out)
        loss += epsilon_penalty * wgan_epsilon

        #loss = tf.reduce_sum(loss)
        #print(global_batch_size)
        #loss = tf.nn.compute_average_loss(loss)
        #reduce average (change when using strategy)
        d_loss = tf.reduce_mean(loss) / n_gpus


    #tf.print('Discrimintator Loss')
    #tf.print(loss)

    #print(grad_total_tape.watched_variables())

    gradients =  grad_total_tape.gradient(d_loss, D.trainable_variables) # model.trainable_variables or  model.trainable_weights



    opt.apply_gradients(zip(gradients, D.trainable_variables)) # model.trainable_variables or  model.trainable_weights

    #there are no labels???
    # if D.output_shapes[1][1] > 0:
    #     with tf.name_scope('LabelPenalty'):
    #         label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
    #         label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
    #         label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
    #         label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
    #     loss += (label_penalty_reals + label_penalty_fakes) * cond_weight


    #apply gradients with optimizes


    return d_loss

def original_Discriminator_loss(G, D, reals, minibatch_size, opt, lod_in=0.0, training_set=None, labels=None,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0,      # Weight of the conditioning terms.
    global_batch_size = 1,
    network_size=256):

    #loss scalling (it is not doing loss scalling) !!!!

    print('Mini-batch size D ' + str(minibatch_size))
    size= int(network_size) 
    print('real shape' + str(reals.shape))
    n_gpus = int(global_batch_size/minibatch_size)
    
    
    #generate noise image
    latents = tf.random.normal([minibatch_size, 3, size, size])
    
    #get labels???
    #labels = training_set.get_random_labels_tf(minibatch_size)

    #needs two gradient tapes because we use the gradeients from the mixed images for the penalties
    with tf.GradientTape(persistent=True) as grad_total_tape:
        #generate fake image
        #inside gradient tape
        with tf.GradientTape() as grad_penalty_tape:
            fake_images_out = G([latents, lod_in], training=True)

            #Apply a gradient penalty
            #needs to be inside a gradient tape to obtain the gradients at the end
            #mixing factors
            mixing_factors = tf.random.uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
            #mixed images
            mixed_images_out = lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
            #get mixed scores
            #mixed_scores_out, mixed_labels_out = fp32(D(mixed_images_out, training=True))
            mixed_scores_out, mixed_labels_out = D([mixed_images_out, lod_in], training=True) #coonvert to f32

            #final mixed loss
            mixed_loss = tf.math.reduce_sum(mixed_scores_out)

        #get mixed gradients
        mixed_grads = fp32(grad_penalty_tape.gradient(mixed_loss, [mixed_images_out])[0])

        # normalize gradients
        mixed_norms = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(mixed_grads), axis=[1,2,3]))

        #apply penalty
        gradient_penalty = tf.math.square(mixed_norms - wgan_target)


        #tf.print(mixed_loss)

        loss = gradient_penalty * (wgan_lambda / (wgan_target**2))

        real_scores_out, real_labels_out = D([reals, lod_in], training=True)
        fake_scores_out, fake_labels_out = D([fake_images_out, lod_in], training=True)
        #real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
        #fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))

        loss += fake_scores_out - real_scores_out


        #epsilon penalty
        epsilon_penalty = tf.math.square(real_scores_out)
        loss += epsilon_penalty * wgan_epsilon

        #loss = tf.reduce_sum(loss)
        #print(global_batch_size)
        loss = tf.reduce_mean(loss) / n_gpus

    #reduce average (change when using strategy)


    gradients =  grad_total_tape.gradient(loss, D.trainable_variables) # model.trainable_variables or  model.trainable_weights
    opt.apply_gradients(zip(gradients, D.trainable_variables)) # model.trainable_variables or  model.trainable_weights

    #there are no labels???
    # if D.output_shapes[1][1] > 0:
    #     with tf.name_scope('LabelPenalty'):
    #         label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
    #         label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
    #         label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
    #         label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
    #     loss += (label_penalty_reals + label_penalty_fakes) * cond_weight


    #apply gradients with optimizes


    return loss

def combined_Discriminator_loss(G, combined_D, encoder, reals, minibatch_size, opt, lod_in=0.0, training_set=None, labels=None,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0,      # Weight of the conditioning terms.
    global_batch_size = 1,
    network_size=256):

    noise_shape = 2048
    
    global_loss_weight = .5
    local_loss_weight = .5

    #loss scalling (it is not doing loss scalling) !!!!
    n_gpus = int(global_batch_size/minibatch_size)
    print('N GPUS: ' +str(n_gpus))
    print('Mini-batch size D ' + str(minibatch_size))
    size= int(network_size / 2) 
    print('real shape' + str(reals.shape))
    
    #get reals:
    real1= reals[:,:, :(size),:(size)]
    real2= reals[:,:, (size):,:(size)]
    real3= reals[:,:, :(size),(size):]
    real4= reals[:,:, (size):, (size):]

    #check axis
    # realsleft = tf.concat([real1, real2], axis=3)
    # reals_in_row = tf.concat([realsleft, real3], axis=3)
    # ae_latents = encoder.encode(reals_in_row)
    
    #generate noise image
    latents = tf.random.normal([minibatch_size, noise_shape])

    ae_latent1 = encoder.encode(real1)
    ae_latent2 = encoder.encode(real2)
    ae_latent3 = encoder.encode(real3)

    #check axis
    ae_latent_left = tf.concat([ae_latent1, ae_latent2], axis=1)
    ae_latents = tf.concat([ae_latent_left, ae_latent3], axis=1)
    

    #needs two gradient tapes because we use the gradeients from the mixed images for the penalties
    with tf.GradientTape() as grad_total_tape:
        #first pass through the discriminator with fake and real data

        #generate fake image
        fake_images_out_small = G([latents, ae_latents, lod_in], training=True)
        fake_image_out_right = tf.concat([real3, fake_images_out_small], axis=2)
        fake_image_out_left = tf.concat([real1, real2], axis=2)
        fake_images_out = tf.concat([fake_image_out_left, fake_image_out_right], axis=3)

        #global Discriminator
        global_real_scores_out = fp32(combined_D.use_global_discriminator(reals, lod_in))
        global_fake_scores_out = fp32(combined_D.use_global_discriminator(fake_images_out, lod_in))
        global_loss = global_fake_scores_out - global_real_scores_out

        #local Discriminator
        local_real_scores_out = fp32(combined_D.use_local_discriminator(real4, lod_in))
        local_fake_scores_out = fp32(combined_D.use_local_discriminator(fake_images_out_small, lod_in))
        local_loss = local_fake_scores_out - local_real_scores_out

        #calculate gradient penalty inside a new gradient tape
        #mixing factors
        mixing_factors = tf.random.uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        #mixed images
        global_mixed_images_out = lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        local_mixed_images_out = lerp(tf.cast(real4, fake_images_out_small.dtype), fake_images_out_small, mixing_factors)

        #global_discriminator
        with tf.GradientTape() as global_grad_penalty_tape:
            global_grad_penalty_tape.watch(global_mixed_images_out)
            #get mixed scores
            global_mixed_scores_out = fp32(combined_D.use_global_discriminator(global_mixed_images_out, lod_in))
            #final mixed loss
            global_mixed_loss = tf.math.reduce_sum(global_mixed_scores_out) #might need to change?
        #get mixed gradients
        global_mixed_grads = fp32(global_grad_penalty_tape.gradient(global_mixed_loss, [global_mixed_images_out])[0])
        # normalize gradients
        global_mixed_norms = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(global_mixed_grads), axis=[1,2,3]))
        #apply penalty
        gradient_penalty = tf.math.square(global_mixed_norms - wgan_target)
        global_loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

        #local_discriminator
        with tf.GradientTape() as local_grad_penalty_tape:
            local_grad_penalty_tape.watch(local_mixed_images_out)
            #get mixed scores
            local_mixed_scores_out = fp32(combined_D.use_local_discriminator(local_mixed_images_out, lod_in))
            #final mixed loss
            local_mixed_loss = tf.math.reduce_sum(local_mixed_scores_out) #might need to change?
        #get mixed gradients
        local_mixed_grads = fp32(local_grad_penalty_tape.gradient(local_mixed_loss, [local_mixed_images_out])[0])
        # normalize gradients
        local_mixed_norms = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(local_mixed_grads), axis=[1,2,3]))
        #apply penalty
        gradient_penalty = tf.math.square(local_mixed_norms - wgan_target)
        local_loss += gradient_penalty * (wgan_lambda / (wgan_target**2))


        #epsilon penalty
        global_epsilon_penalty = tf.math.square(global_real_scores_out)
        global_loss += global_epsilon_penalty * wgan_epsilon

        local_epsilon_penalty = tf.math.square(local_real_scores_out)
        local_loss += local_epsilon_penalty * wgan_epsilon

        
        join_loss = (global_loss_weight * global_loss + local_loss_weight * local_loss)
        d_loss = tf.reduce_mean(join_loss) / n_gpus


    gradients =  grad_total_tape.gradient(d_loss, combined_D.trainable_variables) # model.trainable_variables or  model.trainable_weights

    opt.apply_gradients(zip(gradients, combined_D.trainable_variables)) # model.trainable_variables or  model.trainable_weights

    return d_loss, tf.reduce_mean(global_loss) / n_gpus, tf.reduce_mean(local_loss) / n_gpus

def combined_Discriminator_test_loss(G, combined_D, encoder, reals, minibatch_size, opt, lod_in=0.0, training_set=None, labels=None,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0,      # Weight of the conditioning terms.
    global_batch_size = 1,
    network_size=256):

    noise_shape = 2048
    
    global_loss_weight = .5
    local_loss_weight = .5

    #loss scalling (it is not doing loss scalling) !!!!
    n_gpus = int(global_batch_size/minibatch_size)
    print('N GPUS: ' +str(n_gpus))
    print('Mini-batch size D ' + str(minibatch_size))
    size= int(network_size / 2) 
    print('real shape' + str(reals.shape))
    
    #get reals:
    real1= reals[:,:, :(size),:(size)]
    real2= reals[:,:, (size):,:(size)]
    real3= reals[:,:, :(size),(size):]
    real4= reals[:,:, (size):, (size):]

    #check axis
    # realsleft = tf.concat([real1, real2], axis=3)
    # reals_in_row = tf.concat([realsleft, real3], axis=3)
    # ae_latents = encoder.encode(reals_in_row)
    
    #generate noise image
    latents = tf.random.normal([minibatch_size, noise_shape])

    ae_latent1 = encoder.encode(real1, training_flag=False)
    ae_latent2 = encoder.encode(real2, training_flag=False)
    ae_latent3 = encoder.encode(real3, training_flag=False)

    #check axis
    ae_latent_left = tf.concat([ae_latent1, ae_latent2], axis=1)
    ae_latents = tf.concat([ae_latent_left, ae_latent3], axis=1)
    

    #needs two gradient tapes because we use the gradeients from the mixed images for the penalties
    #with tf.GradientTape() as grad_total_tape:
        #first pass through the discriminator with fake and real data

    #generate fake image
    fake_images_out_small = G([latents, ae_latents, lod_in], training=False)
    fake_image_out_right = tf.concat([real3, fake_images_out_small], axis=2)
    fake_image_out_left = tf.concat([real1, real2], axis=2)
    fake_images_out = tf.concat([fake_image_out_left, fake_image_out_right], axis=3)

    #global Discriminator
    global_real_scores_out = fp32(combined_D.use_global_discriminator(reals, lod_in, training_flag=False))
    global_fake_scores_out = fp32(combined_D.use_global_discriminator(fake_images_out, lod_in, training_flag=False))
    global_loss = global_fake_scores_out - global_real_scores_out

    #local Discriminator
    local_real_scores_out = fp32(combined_D.use_local_discriminator(real4, lod_in, training_flag=False))
    local_fake_scores_out = fp32(combined_D.use_local_discriminator(fake_images_out_small, lod_in, training_flag=False))
    local_loss = local_fake_scores_out - local_real_scores_out

    #calculate gradient penalty inside a new gradient tape
    #mixing factors
    mixing_factors = tf.random.uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
    #mixed images
    global_mixed_images_out = lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
    local_mixed_images_out = lerp(tf.cast(real4, fake_images_out_small.dtype), fake_images_out_small, mixing_factors)

    #global_discriminator
    with tf.GradientTape() as global_grad_penalty_tape:
        global_grad_penalty_tape.watch(global_mixed_images_out)
        #get mixed scores
        global_mixed_scores_out = fp32(combined_D.use_global_discriminator(global_mixed_images_out, lod_in, training_flag=False))
        #final mixed loss
        global_mixed_loss = tf.math.reduce_sum(global_mixed_scores_out) #might need to change?
    #get mixed gradients
    global_mixed_grads = fp32(global_grad_penalty_tape.gradient(global_mixed_loss, [global_mixed_images_out])[0])
    # normalize gradients
    global_mixed_norms = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(global_mixed_grads), axis=[1,2,3]))
    #apply penalty
    gradient_penalty = tf.math.square(global_mixed_norms - wgan_target)
    global_loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    #local_discriminator
    with tf.GradientTape() as local_grad_penalty_tape:
        local_grad_penalty_tape.watch(local_mixed_images_out)
        #get mixed scores
        local_mixed_scores_out = fp32(combined_D.use_local_discriminator(local_mixed_images_out, lod_in, training_flag=False))
        #final mixed loss
        local_mixed_loss = tf.math.reduce_sum(local_mixed_scores_out) #might need to change?
    #get mixed gradients
    local_mixed_grads = fp32(local_grad_penalty_tape.gradient(local_mixed_loss, [local_mixed_images_out])[0])
    # normalize gradients
    local_mixed_norms = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(local_mixed_grads), axis=[1,2,3]))
    #apply penalty
    gradient_penalty = tf.math.square(local_mixed_norms - wgan_target)
    local_loss += gradient_penalty * (wgan_lambda / (wgan_target**2))


    #epsilon penalty
    global_epsilon_penalty = tf.math.square(global_real_scores_out)
    global_loss += global_epsilon_penalty * wgan_epsilon

    local_epsilon_penalty = tf.math.square(local_real_scores_out)
    local_loss += local_epsilon_penalty * wgan_epsilon


    join_loss = (global_loss_weight * global_loss + local_loss_weight * local_loss)
    d_loss = tf.reduce_mean(join_loss) / n_gpus

    return d_loss, tf.reduce_mean(global_loss) / n_gpus, tf.reduce_mean(local_loss) / n_gpus


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def variationa_auto_encoder_loss(cvae, batch, global_batch_size, opt):

    #Join the 3 corner images in a row
    size = 128
    batch1= batch[:,:, :(size),:(size)]
    batch2= batch[:,:, (size):,:(size)]
    batch3= batch[:,:, :(size),(size):]
    batch4= batch[:,:, (size):, (size):]

    corners = [batch1, batch2, batch3, batch4]

    #batchleft = tf.concat([batch1, batch2], axis=3)
    #batchall3 = tf.concat([batchleft, batch3], axis=3)
    #print(tf.shape(batchall3))

    
    for corner in corners:
        with tf.GradientTape() as tape_encoder:
            mean, logvar = cvae.encode(corner)
            z = cvae.reparameterize(mean, logvar)
            reconstruction = cvae.decode(z)
            #mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            #reconstruction_loss = 1000 * mse(batchall3,reconstruction)
            r_w = 1
            kl_w = 1
            reconstruction_loss = tf.math.reduce_mean(tf.math.square(corner - reconstruction), axis = [1,2,3])
            # #reduce mean / reduce  sum ?
            #reconstruction_loss = tf.nn.compute_average_loss(reconstruction_loss, global_batch_size=global_batch_size)
            #reconstruction = tf.math.reduce_sum(reconstruction)
            #add sum tlast 3 elements?
            #kl_loss = -0.5 * tf.math.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))
            kl_loss = - 0.5 * tf.math.reduce_sum(1 + logvar - tf.math.square(mean) - tf.exp(logvar), axis = 1)
            #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            #kl_loss = tf.nn.compute_average_loss(kl_loss, global_batch_size=global_batch_size)
            total_loss = r_w * reconstruction_loss + kl_w * kl_loss

            final_loss = tf.nn.compute_average_loss(total_loss, global_batch_size=global_batch_size)

        encoder_grads = tape_encoder.gradient(final_loss, cvae.trainable_variables)

    
        opt.apply_gradients(zip(encoder_grads, cvae.trainable_variables))

    return tf.nn.compute_average_loss(reconstruction_loss, global_batch_size=global_batch_size), tf.nn.compute_average_loss(kl_loss, global_batch_size=global_batch_size)

def beta_auto_encoder_loss(cvae, batch, global_batch_size, opt, iterat, desintangled=False):

    #Join the 3 corner images in a row
    size = 128
    batch1= batch[:,:, :(size),:(size)]
    batch2= batch[:,:, (size):,:(size)]
    batch3= batch[:,:, :(size),(size):]
    batch4= batch[:,:, (size):, (size):]

    corners = [batch1, batch2, batch3, batch4]

    batch_size = tf.shape(batch1)[0] 

    r_w = 1
    kl_w = 0.005

    #batchleft = tf.concat([batch1, batch2], axis=3)
    #batchall3 = tf.concat([batchleft, batch3], axis=3)
    #print(tf.shape(batchall3))

    
    for corner in corners:
        with tf.GradientTape() as tape_encoder:
            mean, logvar = cvae.encode(corner)
            z = cvae.reparameterize(mean, logvar)
            reconstruction = cvae.decode(z)
            #mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            #reconstruction_loss = 1000 * mse(batchall3,reconstruction)
            reconstruction_loss = tf.math.reduce_mean(tf.math.square(corner - reconstruction), axis = [1,2,3])
            # #reduce mean / reduce  sum ?
            #reconstruction_loss = tf.nn.compute_average_loss(reconstruction_loss, global_batch_size=global_batch_size)
            #reconstruction = tf.math.reduce_sum(reconstruction)
            #add sum tlast 3 elements?
            #kl_loss = -0.5 * tf.math.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))
            kl_loss = tf.math.reduce_mean(- 0.5 * tf.math.reduce_sum(1 + logvar - tf.math.square(mean) - tf.exp(logvar), axis = 1))
            #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            #kl_loss = tf.nn.compute_average_loss(kl_loss, global_batch_size=global_batch_size)

            if desintangled:
                total_loss = reconstruction_loss + cvae.beta * kl_w * kl_loss
            else:
                C = tf.clip_by_value(cvae.C_max/cvae.C_stop_iter * iterat, 0., cvae.C_max)
                total_loss = reconstruction_loss + cvae.gamma * kl_w* tf.math.abs(kl_loss - C)

            final_loss = total_loss#tf.nn.compute_average_loss(total_loss, global_batch_size=global_batch_size)

        encoder_grads = tape_encoder.gradient(final_loss, cvae.trainable_variables)

    
        opt.apply_gradients(zip(encoder_grads, cvae.trainable_variables))

    return tf.nn.compute_average_loss(reconstruction_loss, global_batch_size=global_batch_size), kl_loss / tf.cast((global_batch_size / batch_size), dtype=tf.float32)#tf.nn.compute_average_loss(kl_loss, global_batch_size=global_batch_size)

def wasserstein_auto_encoder_loss(cvae, batch, global_batch_size, opt, size, kernel_type = 'rbf'):

    init_reg_weight = 100

    #Join the 3 corner images in a row
    #size = 128
    batch1= batch[:,:, :(size),:(size)]
    batch2= batch[:,:, (size):,:(size)]
    batch3= batch[:,:, :(size),(size):]
    batch4= batch[:,:, (size):, (size):]

    corners = [batch1, batch2, batch3, batch4]

    #batchleft = tf.concat([batch1, batch2], axis=3)
    #batchall3 = tf.concat([batchleft, batch3], axis=3)
    #print(tf.shape(batchall3))

    r_weight = 1
    mmd_weight = 1
    angle_weight= 1

    
    for corner in corners:
        with tf.GradientTape() as tape_encoder:
            #mean, logvar = cvae.encode(corner) #do I use mean and logvar?
            #z = cvae.reparameterize(mean, logvar)
            z = cvae.encode(corner) #do I use mean and logvar?
            reconstruction = cvae.decode(z)

            #reg_weight
            batch_size = tf.shape(corner)[0] 
            bias_corr = batch_size *  (batch_size - 1)
            reg_weight = init_reg_weight / bias_corr
            reg_weight = tf.cast(reg_weight, dtype=tf.float32)

            #mse
            reconstruction_loss = r_weight * tf.math.reduce_mean(tf.math.square(corner - reconstruction), axis = [1,2,3])
           
            #angle between the RGB triplet
            #https://arxiv.org/pdf/1504.04548.pdf
            #https://arxiv.org/pdf/1906.01340.pdf

            # numerator = tf.math.reduce_sum(tf.math.multiply(corner, reconstruction), axis = 1)
            # denominator = tf.math.multiply(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(corner), axis = 1)) ,tf.math.sqrt(tf.math.reduce_sum(tf.math.square(reconstruction), axis = 1))) #+ 1e-8

            # tf.print('a')
            #tf.print(numerator)
            # tf.print('b')
            #tf.print(denominator)

            #tf.print(tf.math.reduce_mean(tf.math.acos(tf.clip_by_value( (numerator / denominator ), -1 + 1e-7, 1 - 1e-7 )),axis=[1,2]))

            #tf acos has problems with the boundaries, gradients go to infinite
            #rgbangle = angle_weight * tf.math.reduce_mean(tf.math.acos(tf.clip_by_value( (numerator / denominator ), -1 + 1e-7, 1 - 1e-7 )),axis=[1,2])
            
            #tf.print(rgbangle)
            #tf.print(tf.nn.compute_average_loss(rgbangle, global_batch_size=global_batch_size))

            #mmd
            prior_z = tf.random.normal(tf.shape(z))

            prior_z__kernel = compute_kernel(prior_z, prior_z, kernel_type = kernel_type)
            z__kernel = compute_kernel(z, z, kernel_type = kernel_type)
            priorz_z__kernel = compute_kernel(prior_z, z, kernel_type = kernel_type)


            prior_z__kernel_loss = reg_weight * tf.math.reduce_mean(prior_z__kernel)
            z__kernel_loss = reg_weight * tf.math.reduce_mean(z__kernel)
            priorz_z__kernel_loss = 2 * reg_weight * tf.math.reduce_mean(priorz_z__kernel)

            mmd_loss = mmd_weight * (prior_z__kernel_loss + z__kernel_loss - priorz_z__kernel_loss)

            #kl_loss = - 0.5 * tf.math.reduce_sum(1 + logvar - tf.math.square(mean) - tf.exp(logvar), axis = 1)
            #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            #kl_loss = tf.nn.compute_average_loss(kl_loss, global_batch_size=global_batch_size)
            total_loss =  (reconstruction_loss + mmd_loss) / tf.cast((global_batch_size / batch_size), dtype=tf.float32)

            final_loss = total_loss #tf.nn.compute_average_loss(total_loss, global_batch_size=global_batch_size)

        encoder_grads = tape_encoder.gradient(final_loss, cvae.trainable_variables)
    
        opt.apply_gradients(zip(encoder_grads, cvae.trainable_variables))

        #tf.print(tf.nn.compute_average_loss(reconstruction_loss, global_batch_size=global_batch_size))
        #tf.print(mmd_loss / tf.cast((global_batch_size / batch_size), dtype=tf.float32))
        # tf.print(tf.nn.compute_average_loss(numerator, global_batch_size=global_batch_size))
        # tf.print(tf.nn.compute_average_loss(denominator, global_batch_size=global_batch_size))
        # tf.print(tf.nn.compute_average_loss(rgbangle, global_batch_size=global_batch_size))

    return tf.nn.compute_average_loss(reconstruction_loss, global_batch_size=global_batch_size), mmd_loss / tf.cast((global_batch_size / batch_size), dtype=tf.float32)#, tf.nn.compute_average_loss(rgbangle, global_batch_size=global_batch_size)#tf.nn.compute_average_loss(mmd_loss, global_batch_size=global_batch_size)


def compute_kernel(x1_o, x2_o, z_var=2., kernel_type = 'rbf'):
    # Convert the tensors into row and column vectors
    D = tf.shape(x1_o)[1] 
    N = tf.shape(x1_o)[0] 


    x1 = tf.expand_dims(x1_o, axis=-2) # Make it into a column tensor
    x2 = tf.expand_dims(x2_o, axis=-3) # Make it into a row tensor

    """
    Usually the below lines are not required, especially in our case,
    but this is useful when x1 and x2 have different sizes
    along the 0th dimension.
    """
    x1 = tf.broadcast_to(x1, [N, N, D])
    x2 = tf.broadcast_to(x2, [N, N, D])


    #compute rbf

    if kernel_type == 'rbf':
        z_dim = tf.cast(tf.shape(x2)[-1], dtype=tf.float32) #what should this be C or W
        sigma = 2. * z_dim * z_var

        result = tf.math.exp(-tf.math.reduce_mean( tf.math.pow((x1 - x2),2) , axis=-1) / sigma)

    elif kernel_type == 'imq':
        eps = 1e-7
        z_dim = tf.cast(tf.shape(x2)[-1], dtype=tf.float32)
        C = 2 * z_dim * z_var
        kernel = C / (eps + C + tf.math.reduce_sum( tf.math.pow((x1 - x2),2) , axis=-1))
        # Exclude diagonal elements
        result = tf.math.reduce_mean(kernel) -  tf.math.reduce_mean(tf.linalg.diag_part(kernel))

    return result

    # if self.kernel_type == 'rbf':
    #     result = self.compute_rbf(x1, x2)
    # elif self.kernel_type == 'imq':
    #     result = self.compute_inv_mult_quad(x1, x2)
    # else:
    #     raise ValueError('Undefined kernel type.')

    #return result


def Beta_TC_auto_encoder_loss(cvae, batch, global_batch_size, opt, iterat, desintangled=False):

    #Join the 3 corner images in a row
    size = 128
    batch1= batch[:,:, :(size),:(size)]
    batch2= batch[:,:, (size):,:(size)]
    batch3= batch[:,:, :(size),(size):]
    batch4= batch[:,:, (size):, (size):]

    corners = [batch1, batch2, batch3, batch4]

    batch_size = 32#float( tf.shape(batch1)[0])

    r_w = 1
    kl_w = 1

    #batchleft = tf.concat([batch1, batch2], axis=3)
    #batchall3 = tf.concat([batchleft, batch3], axis=3)
    #print(tf.shape(batchall3))

    d_w = 0.005

    
    for corner in corners:
        with tf.GradientTape() as tape_encoder:
            mean, logvar = cvae.encode(corner)
            z = cvae.reparameterize(mean, logvar)
            reconstruction = cvae.decode(z)
            
            reconstruction_loss = tf.math.reduce_mean(tf.math.square(corner - reconstruction), axis = [1,2,3]) #reduction sum?

            log_q_zx = tf.math.reduce_sum(log_density_gaussian(z, mean, logvar), axis=1)

            zeros = tf.zeros(tf.shape(z))
            log_p_z = tf.math.reduce_sum(log_density_gaussian(z, zeros, zeros), axis=1)

            latent_dim = tf.shape(z)[1] 
            a = tf.reshape(z,(batch_size, 1, latent_dim))
            b = tf.reshape(mean, (1, batch_size, latent_dim))
            c = tf.reshape(logvar, (1, batch_size, latent_dim))
            mat_log_q_z = log_density_gaussian(a, b , c)


            dataset_size = (1. / d_w) * batch_size # dataset size
            strat_weight = (dataset_size - batch_size + 1.) / (dataset_size * (batch_size - 1.))
            importance_weights = np.full((batch_size, batch_size), 1.)
            #importance_weights = tf.fill((batch_size, batch_size), 1 / (batch_size -1))
            np.reshape(importance_weights, (-1))[::batch_size] = 1 / batch_size
            np.reshape(importance_weights, (-1))[:batch_size] = 1 / strat_weight
            importance_weights[batch_size - 2, 0] = strat_weight
            log_importance_weights = np.log(importance_weights)
            log_importance_weights = np.float32(log_importance_weights)

            mat_log_q_z += tf.reshape(log_importance_weights, (batch_size, batch_size, 1))

            # log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
            # log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

            log_q_z = tf.math.reduce_logsumexp(tf.math.reduce_sum( mat_log_q_z, axis = 2), axis = 1)
            log_prod_q_z = tf.math.reduce_sum(tf.math.reduce_logsumexp( mat_log_q_z, axis = 1), axis = 1)

            mi_loss  = tf.reduce_mean(log_q_zx - log_q_z)
            tc_loss = tf.reduce_mean(log_q_z - log_prod_q_z)
            kld_loss = tf.reduce_mean(log_prod_q_z - log_p_z)

            anneal_rate = tf.math.minimum(0 + 1 * iterat / cvae.anneal_steps, 1)

            final_loss = reconstruction_loss + cvae.alpha * mi_loss + kl_w * (cvae.beta * tc_loss + anneal_rate * cvae.gamma * kld_loss)


        encoder_grads = tape_encoder.gradient(final_loss, cvae.trainable_variables)

    
        opt.apply_gradients(zip(encoder_grads, cvae.trainable_variables))

    return tf.nn.compute_average_loss(reconstruction_loss, global_batch_size=global_batch_size), kld_loss / tf.cast((global_batch_size / batch_size), dtype=tf.float32)#tf.nn.compute_average_loss(kl_loss, global_batch_size=global_batch_size)


def log_density_gaussian(x, mu, logvar):
    #Computes the log pdf of the Gaussian with parameters mu and logvar at x
    
    norm = - 0.5 * (math.log(2 * math.pi) + logvar)
    log_density = norm - 0.5 * (tf.math.pow((x - mu), 2) * tf.math.exp(-logvar))
    return log_density