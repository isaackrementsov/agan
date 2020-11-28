import time
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers

from utils import to_image, to_video

class AGAN:

    def __init__(self, noise_size, batch_size):
        self.batch_size = batch_size
        self.noise_size = noise_size

    def initialize(self, generator, discriminator):
        self.G = generator
        self.D = discriminator

        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)

    def new(self):
        generator = keras.Sequential()
        generator.add(layers.Dense(25*25*180, use_bias=False, input_shape=(self.noise_size,)))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())
        generator.add(layers.Reshape((25,25,180)))
        generator.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())
        generator.add(layers.Conv2DTranspose(64, (5,5), strides=(1,1), padding='same', use_bias=False))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())
        generator.add(layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', use_bias=False))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())
        generator.add(layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))

        discriminator = keras.Sequential()
        discriminator.add(layers.Dropout(0.1))
        discriminator.add(layers.Conv2D(16, (2,2), strides=(2,2), padding='same', input_shape=(100,100,3)))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Conv2D(32, (2,2), strides=(2,2), padding='same'))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Conv2D(64, (2,2), strides=(2,2), padding='same'))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Conv2D(128, (2,2), strides=(2,2), padding='same'))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Flatten())
        discriminator.add(layers.Dense(32))
        discriminator.add(layers.Dense(16, activation='relu'))
        discriminator.add(layers.Dense(8,  activation='relu'))
        discriminator.add(layers.Dense(1))

        self.optimizer_D = keras.optimizers.Adam(1e-4)
        self.optimizer_G = keras.optimizers.Adam(1e-4)

        self.restored = False

        self.initialize(generator, discriminator)

    def restore(self):
        generator = keras.models.load_model('Generator', compile=False)
        self.optimizer_G = generator.optimizer

        discriminator = keras.models.load_model('Discriminator', compile=False)
        self.optimizer_D = discriminator.optimizer

        self.restored = True

        self.initialize(generator, discriminator)

    def save(self):
        self.G.optimizer = self.optimizer_G
        self.G.save('Generator')

        self.D.optimizer = self.optimizer_D
        self.D.save('Discriminator')

    def generate_examples(self, name):
        generated = self.G(tf.random.normal([16, self.noise_size]), training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(generated.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((generated[i,:,:] + 1)/2)
            plt.axis('off')

        plt.savefig('./examples/' + name  + '.png')
        plt.close('all')

    def generate_image(self, name):
        generated = self.G(tf.random.normal([1, self.noise_size]), training=False)
        to_image(generated).save(name + '.jpg')

    def generate_animation(self, name, frames, samples):
        noise = tf.random.normal([samples, 100])
        interpolated = tf.zeros([0, 100])
        frames_per_sample = frames//samples

        for i in range(samples - 1):
             start_vec = noise[i]
             end_vec = noise[i + 1]
             dz = (end_vec - start_vec)/frames_per_sample

             for j in range(frames_per_sample):
                 interpolated = tf.concat([interpolated, [start_vec + j*dz]], 0)

        generated_frames = self.G(interpolated, training=False)
        to_video(generated_frames, name)

    def get_offset(self, example_interval):
        if self.restored:
            filenames = os.listdir('examples/')

            if len(filenames) > 0:
                last = filenames[-1].split('.png')[0].split('epoch')[-1]
                return int(last) + example_interval
            else:
                return 0
        else:
            return 0

    def noise(self):
        return tf.random.normal([self.batch_size, self.noise_size])

    # Loss from image being noisy (high-frequency "jumps")
    def denoise_loss(self, images, shift):
        # Get how much the image varies due to pixels shifted horizontally and vertically
        x_var = images[:,:,shift:,:] - images[:,:,:-shift,:]
        y_var = images[:,shift:,:,:] - images[:,:-shift,:,:]

        return tf.reduce_sum(tf.abs(x_var)) + tf.reduce_sum(tf.abs(y_var))

    def loss_D(self, real, fake):
        real_loss = self.bce(tf.ones_like(real), real)
        generated_loss = self.bce(tf.zeros_like(fake), fake)

        return real_loss + generated_loss

    def loss_G(self, fake):
        return self.bce(tf.ones_like(fake), fake)

    def train(self, dataset, epochs, example_interval, save_interval):
        example_offset = self.get_offset(example_interval)

        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            if (epoch + 1) % example_interval == 0 or epoch == 0:
                self.generate_examples('epoch' + str(epoch + example_offset + 1))

            if (epoch + 1) % save_interval == 0 or epoch == 0:
                self.save()

            print('Epoch #{} took {} seconds'.format(epoch + 1, time.time() - start))

    @tf.function
    def train_step(self, images):
        noise = self.noise()

        with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:
            generated = self.G(noise, training=True)
            real = self.D(images, training=True)
            fake = self.D(generated, training=True)

            loss_G = self.loss_G(fake)
            loss_D = self.loss_D(real, fake)

        gradients_G = tape_G.gradient(loss_G, self.G.trainable_variables)
        gradients_D = tape_D.gradient(loss_D, self.D.trainable_variables)

        self.optimizer_G.apply_gradients(zip(gradients_G, self.G.trainable_variables))
        self.optimizer_D.apply_gradients(zip(gradients_D, self.D.trainable_variables))
