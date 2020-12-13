import time
import os
import sys
import matplotlib.pyplot as plt
from random import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from keras import layers

from model.discriminator import Discriminator
from model.generator import Generator, StyleMapper
from utils import to_image, to_video, to_frames

class AGAN:

    def __init__(self, batch_size, resolution, mix_prob=0.5):
        self.batch_size = batch_size
        self.resolution = resolution
        self.mix_prob = mix_prob
        
    def initialize(self, style_mapper, generator, discriminator):
        style_mapper.model.optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5)
        generator.model.optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5)
        discriminator.model.optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5)

        self.S = style_mapper
        self.G = generator
        self.D = discriminator

        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)

    def new(self):
        print('Creating a new model...')

        style_mapper = StyleMapper()
        generator = Generator(self.resolution, self.batch_size)
        discriminator = Discriminator(self.resolution)

        self.restored = False

        self.initialize(style_mapper, generator, discriminator)

    def restore(self):
        print('Restoring model...')

        style_mapper = StyleMapper()
        style_mapper.restore()

        generator = Generator(self.resolution, self.batch_size)
        generator.restore()

        discriminator = Discriminator(self.resolution)
        discriminator.restore()

        self.restored = True

        self.initialize(style_mapper, generator, discriminator)

    def save(self):
        self.G.save()
        self.S.save()
        self.D.save()
        
    def generate_examples(self, name):
        styles, noise = self.get_generator_inputs(self.batch_size)
        generated = self.G([styles, noise])
        
        fig = plt.figure(figsize=(16,16))

        for i in range(min(16, generated.shape[0])):
            plt.subplot(4, 4, i + 1)
            plt.imshow(tf.clip_by_value((generated[i,:,:] + 1)/2, clip_value_min=0, clip_value_max=1))
            plt.axis('off')

        plt.savefig('./examples/' + name  + '.png')
        plt.close('all')

    def generate_image(self, name):
        styles, noise = get_generator_inputs(1)
        generated = self.G([styles, noise], training=False)

        to_image(generated).save(name + '.jpg')

    def generate_animation(self, name, frames, samples):
        latent_points = tf.random.normal([samples, self.S.latent_size])
        interpolated = tf.zeros([0, self.S.latent_size])
        frames_per_sample = frames//samples

        for i in range(samples - 1):
             start_vec = latent_points[i]
             end_vec = latent_points[i + 1]
             dz = (end_vec - start_vec)/frames_per_sample

             for j in range(frames_per_sample):
                 interpolated = tf.concat([interpolated, [start_vec + j*dz]], 0)

        interpolated_batches = tf.split(interpolated, num_or_size_splits=samples)
        shape = (0,0,0)

        for i in range(len(interpolated_batches)):
            styles, noise = self.custom_generator_inputs(len(interpolated_batches[i]), interpolated_batches[i])
            generated_frames = self.G([styles, noise], training=False)

            to_frames(generated_frames, i*frames_per_sample)
            if i == 0: shape = generated_frames[0].shape

        to_video(name)

    def get_offset(self):
        if self.restored:
            filenames = os.listdir('examples/')

            if len(filenames) > 0:
                numbers = [int(name.split('.png')[0].split('epoch')[-1]) for name in filenames]
                numbers.sort()

                last = max(numbers)
                return last
            else:
                return 0
        else:
            return 0

    def get_generator_inputs(self, batches):
        n_blocks = self.G.n_style_blocks + 1
        z = lambda: tf.random.normal([batches, self.S.latent_size])
        
        if random() >= self.mix_prob:
            print('mix')
            d = int(random()*n_blocks)
            z1 = [z()]*d
            z2 = [z()]*(n_blocks - d)

            z_points = z1 + [] + z2
        else:
            print('not mixed')
            z_points = [z()]*n_blocks
        
        return self.custom_generator_inputs(batches, z_points)

    def custom_generator_inputs(self, batches, latent_points):
        w_points = []
        for z in latent_points:
            w_points.append(self.S(z))

        noise = tf.random.uniform([batches, self.resolution, self.resolution, 1])

        return w_points, noise

    # Loss from image being noisy (high-frequency "jumps")
    def denoise_loss(self, images, shift):
        # Get how much the image varies due to pixels shifted horizontally and vertically
        x_var = images[:,:,shift:,:] - images[:,:,:-shift,:]
        y_var = images[:,shift:,:,:] - images[:,:-shift,:,:]

        return tf.reduce_sum(tf.abs(x_var)) + tf.reduce_sum(tf.abs(y_var))

    def loss_D(self, real, fake):
        real_loss = K.relu(1 + real)
        generated_loss = K.relu(1 - fake)

        return K.mean(real_loss + generated_loss)

    def loss_G(self, fake):
        return K.mean(fake)

    def loss_PL(self):
        return 0
    
    def train(self, dataset, epochs, example_interval, save_interval):
        example_offset = self.get_offset()

        for epoch in range(epochs):
            start = time.time()

            for i in range(len(dataset)):
                last = (i == len(dataset) - 1)
                self.train_step(dataset[i], last)

            if (epoch + 1) % example_interval == 0 or epoch == 0:
                self.generate_examples('epoch' + str(epoch + example_offset + 1))

            if (epoch + 1) % save_interval == 0 or epoch == 0:
                self.save()

            print('Epoch #{} took {} seconds'.format(epoch + 1, time.time() - start))

    @tf.function
    def train_step(self, images, last):
        with tf.GradientTape() as tape_S, tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:
            styles, noise = self.get_generator_inputs(self.batch_size)

            generated = self.G([styles, noise])
            real = self.D(images)
            fake = self.D(generated)

            loss_G = self.loss_G(fake)
            loss_D = self.loss_D(real, fake)

            if last:
                print('Generator loss:')
                tf.print(loss_G, output_stream=sys.stdout)
                print('Discriminator loss:')
                tf.print(loss_D, output_stream=sys.stdout)

        gradients_S = tape_S.gradient(loss_G, self.S.model.trainable_variables)
        gradients_G = tape_G.gradient(loss_G, self.G.model.trainable_variables)
        gradients_D = tape_D.gradient(loss_D, self.D.model.trainable_variables)

        self.S.model.optimizer.apply_gradients(zip(gradients_S, self.S.model.trainable_variables))
        self.G.model.optimizer.apply_gradients(zip(gradients_G, self.G.model.trainable_variables))
        self.D.model.optimizer.apply_gradients(zip(gradients_D, self.D.model.trainable_variables))
