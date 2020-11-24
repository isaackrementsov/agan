import time
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers

class AGAN:

    def __init__(self, noise_size, batch_size):
        self.batch_size = batch_size
        self.noise_size = noise_size

        generator = keras.Sequential()
        generator.add(layers.Dense(25*25*180, use_bias=False, input_shape=(self.noise_size,)))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())
        generator.add(layers.Reshape((25,25,180)))
        generator.add(layers.Conv2DTranspose(90, (5,5), strides=(1,1), padding='same', use_bias=False))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())
        generator.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())
        generator.add(layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))

        self.G = generator

        discriminator = keras.Sequential()
        discriminator.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=(100,100,3)))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Dropout(0.3))
        discriminator.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Dropout(0.3))
        discriminator.add(layers.Flatten())
        discriminator.add(layers.Dense(1))

        self.D = discriminator

        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer_D = keras.optimizers.Adam(1e-4)
        self.optimizer_G = keras.optimizers.Adam(1e-4)

        self.checkpoint_dir = './checkpoints'
        self.checkpoint_path = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint()
        self.checkpoint.mapped = {
            'optimizer_G': self.optimizer_G,
            'optimizer_D': self.optimizer_D,
            'G': self.G,
            'D': self.D
        }

    def save_checkpoint(self):
        self.checkpoint.save(file_prefix=self.checkpoint_path)

    def restore_latest(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def generate_examples(self, name):
        generated = self.G(tf.random.normal([16, self.noise_size]), training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(generated.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((generated[i,:,:] + 1)/2)
            plt.axis('off')

        plt.savefig('./examples/' + name  + '.png')

    def noise(self):
        return tf.random.normal([self.batch_size, self.noise_size])

    def loss_D(self, real, fake):
        real_loss = self.bce(tf.ones_like(real), real)
        generated_loss = self.bce(tf.zeros_like(fake), fake)

        return real_loss + generated_loss

    def loss_G(self, fake):
        return self.bce(tf.ones_like(fake), fake)

    def train(self, dataset, epochs, save_interval, batch_no):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            if (epoch + 1) % save_interval == 0 or epoch == 0:
                self.save_checkpoint()
                self.generate_examples('epoch' + str(epoch + 1))

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
