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
from model.generator import Generator
from utils import to_image, to_video, lerp

class AGAN:

    def __init__(self, batch_size, resolution, mix_prob=0.5):
        self.batch_size = batch_size
        self.resolution = resolution
        self.mix_prob = mix_prob

    def initialize(self, generator, discriminator):
        self.G = generator
        self.D = discriminator

        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)

    def new(self):
        print('Creating a new model...')

        generator = Generator(self.resolution, depth=6)
        discriminator = Discriminator(self.resolution, depth=6)

        lr = 1e-4
        b1 = 0.5
        generator.model.optimizer = keras.optimizers.Adam(lr, beta_1=b1)
        discriminator.model.optimizer = keras.optimizers.Adam(lr, beta_1=b1)

        self.restored = False

        self.initialize(generator, discriminator)

    def restore(self):
        print('Restoring model...')

        generator = Generator(self.resolution, self.batch_size)
        generator.restore()

        discriminator = Discriminator(self.resolution)
        discriminator.restore()

        self.restored = True

        self.initialize(generator, discriminator)

    def save(self):
        self.G.save()
        self.D.save()

    def generate_examples(self, name):
        Goz = self.G(self.example_z)
        fig = plt.figure(figsize=(16,16))

        for i in range(min(16, Goz.shape[0])):
            plt.subplot(4, 4, i + 1)
            plt.imshow(tf.clip_by_value((Goz[i,:,:] + 1)/2, clip_value_min=0, clip_value_max=1))
            plt.axis('off')

        plt.savefig('./examples/' + name  + '.png')
        plt.close('all')

    def generate_image(self, name):
        z = self.get_latent_inputs(self.batch_size)
        Goz = self.G(z, training=False)

        to_image(Goz[0]).save(name + '.jpg')

    def generate_from_mapper(self, name, latent_points):
        # Insert batch dimension to latent points
        z = tf.expand_dims(latent_points, axis=1)
        Goz = self.G(z, training=False)

        to_image(Goz[0]).save(name + '.jpg')

    def generate_animation(self, name, frames, samples):
        frames_per_sample = frames//samples

        z1 = self.get_latent_inputs(self.batch_size)

        for i in range(samples):
            z2 = self.get_latent_inputs(self.batch_size)

            for j in range(frames_per_sample):
                # Linear interpolation of the two latent points
                z = lerp(z1, z2, j, frames_per_sample)
                Goz = self.G(z, training=False)
                # Frame number for putting animation together
                frame_no = i*frames_per_sample + j
                # Save the output for later processing (converting a batch of outputs directly to video can result in OOM)
                to_image(Goz[0]).save('./frames/frame' + str(frame_no) + '.jpg')

            z1 = z2

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

    def get_latent_inputs(self, batches):
        return tf.random.normal([batches, self.G.z_length])

    # Loss from image being noisy (high-frequency "jumps")
    def denoise_loss(self, images, shift):
        # Get how much the image varies due to pixels shifted horizontally and vertically
        x_var = images[:,:,shift:,:] - images[:,:,:-shift,:]
        y_var = images[:,shift:,:,:] - images[:,:-shift,:,:]

        return tf.reduce_sum(tf.abs(x_var)) + tf.reduce_sum(tf.abs(y_var))

    def loss_D(DoGoz, Dox):
        real_loss = self.bce(tf.ones_like(Dox), Dox)
        generated_loss = self.bce(tf.zeros_like(DoGoz), DoGoz)

        return real_loss + generated_loss

    def loss_G(DoGoz):
        return self.bce(tf.ones_like(DoGoz), DoGoz)

    def train(self, dataset, epochs, example_interval, save_interval):
        example_offset = self.get_offset()

        self.example_z = self.get_latent_inputs(self.batch_size)

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
    def train_step(self, x, last):
        with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:
            # Get noise vector
            z = self.get_latent_inputs(self.batch_size)

            Goz = self.G(z)
            Dox = self.D(x)
            DoGoz = self.D(Goz)

            loss_D = AGAN.loss_D(DoGoz, Dox)
            loss_G = AGAN.loss_G(DoGoz)

            if last:
                tf.print(loss_G, output_stream=sys.stdout)
                tf.print(loss_D, output_stream=sys.stdout)

        gradients_D = tape_D.gradient(loss_D, self.D.model.trainable_variables)
        self.D.model.optimizer.apply_gradients(zip(gradients_D, self.D.model.trainable_variables))

        gradients_G = tape_G.gradient(loss_G, self.G.model.trainable_variables)
        self.G.model.optimizer.apply_gradients(zip(gradients_G, self.G.model.trainable_variables))
