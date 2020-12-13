import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Discriminator:

    def __init__(self, max_size, depth=3, restore=False, downsample_size=2):
        self.max_size = max_size
        self.depth = depth
        self.downsample_size = downsample_size

        if restore:
            self.model = self.restore()
        else:
            self.model = self.new()

    def block(self, x, filters, avg_pooling=True):

        r = layers.Conv2D(filters, (1,1), kernel_initializer='he_uniform')(x)

        x = layers.Conv2D(filters, (3,3), padding='same', kernel_initializer='he_uniform')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(filters, (3,3), padding='same', kernel_initializer='he_uniform')(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.add([x,r])

        if avg_pooling:
            x = layers.AveragePooling2D(self.downsample_size)(x)

        return x

    def new(self):
        min_size = 4
        n_layers = int(np.log(min_size/self.max_size)/np.log(self.downsample_size))

        x = keras.Input([self.max_size, self.max_size, 3])
        y = x

        filters = self.depth

        for i in range(n_layers):
            filters *= 2
            y = self.block(y, filters)

        y = self.block(y, filters, avg_pooling=False)

        y = layers.Flatten()(y)
        y = layers.Dense(1, kernel_initializer='he_uniform')(y)

        return keras.Model(inputs=x, outputs=y)

    def __call__(self, inputs):
        return self.model(inputs)

    def restore(self):
        return keras.models.load_model('Discriminator', compile=False)

    def save(self):
        return keras.models.save_model(self.model, 'Discriminator')
