# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf
import networks2
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

def Generator_loss(G, D, reals, minibatch_size, opt=None, lod_in=0.0, training_set=None, cond_weight = 1.0, network_size=256, global_batch_size = 1): # Weight of the conditioning term.
    print('Mini-batch size G ' + str(minibatch_size))
    size= int(network_size / 2)  
    n_gpus = int(global_batch_size/minibatch_size)  
    # Conditional GAN Loss
    real1= reals[:,:, :(size),:(size)]
    real2= reals[:,:, (size):,:(size)]
    real3= reals[:,:, :(size),(size):]
    real4= reals[:,:, :(size), :(size)]
    
   
    #generate fakes
    latents = tf.random.normal([minibatch_size, 3, size, size])
    left = tf.concat([real1, real2], axis=2)
    right = tf.concat([real3, latents], axis=2)
    lat_and_cond = tf.concat([left, right], axis=3)

    # file_test = pickle.load( open( 'test_objects.pkl', 'rb' ) )

    # lat_and_cond = file_test['lat_and_cond']

    # lat_and_cond = tf.convert_to_tensor(lat_and_cond)

    
    print('lat_and_cond : ' + str(lat_and_cond))
    #tf.print(lat_and_cond)


    #get labels  (are there any labels???)
    #labels = training_set.get_random_labels_tf(minibatch_size)


    #fake_images_out_small = G.get_output_for(lat_and_cond, labels, is_training=True)

    with tf.GradientTape() as tape:
        fake_images_out_small = G([lat_and_cond, lod_in], training=True)
        #tf.print(fake_images_out_small)
        fake_image_out_right = tf.concat([real3, fake_images_out_small], axis=2)
        fake_image_out_left = tf.concat([real1, real2], axis=2)
        fake_images_out = tf.concat([fake_image_out_left, fake_image_out_right], axis=3)
        fake_scores_out= fp32(D([fake_images_out, lod_in], training=True))
        #tf.print(fake_scores_out)
        loss = -fake_scores_out 


        #loss = tf.reduce_mean(loss)
        #tf.print(loss)
        #loss = tf.nn.compute_average_loss(loss)#, global_batch_size=global_batch_size)
        g_loss = tf.reduce_mean(loss) / n_gpus


        #tf.print(loss)

    #tf.print('Generator Loss')
    #tf.print(n_gpus)
    #tf.print(loss)

    #add label penalty (are there any labels???)
    #if D.output_shapes[1][1] > 0:
    #    with tf.name_scope('LabelPenalty'):
    #        label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
    #    loss += label_penalty_fakes * cond_weight

    #tf.print(var.name for var in tape.watched_variables())
        
    gradients =  tape.gradient(g_loss, G.trainable_variables) # model.trainable_variables or  model.trainable_weights
    #for var in G.trainable_variables:
        #tf.print(var.name)
    opt.apply_gradients(zip(gradients, G.trainable_variables)) # model.trainable_variables or  model.trainable_weights
    
    
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