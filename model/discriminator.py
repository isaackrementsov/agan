import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K

class Clip(keras.constraints.Constraint):

    def __init__(self, clip_value_max, clip_value_min):
        # Set clip values
        self.clip_value_max = clip_value_max
        self.clip_value_min = clip_value_min

    def __call__(self, weights):
        # Keep weights within [clip_value_min, clip_value_max]
        return K.clip(weights, self.clip_value_min, self.clip_value_min)

    def get_config(self):
        return {'clip_value_max': self.clip_value_max, 'clip_value_min': self.clip_value_min}


class Discriminator:

    def __init__(self, max_size, depth=3, restore=False, downsample_size=2, clip_range=(0.01,0.01)):
        self.max_size = max_size
        self.depth = depth
        self.downsample_size = downsample_size
        # Constrain weights to a "box" according to the WGAN paper
        self.clip_constraint = Clip(*clip_range)

        if restore:
            self.model = self.restore()
        else:
            self.model = self.new()

    def block(self, x, filters, avg_pooling=True):

        r = layers.Conv2D(filters, (1,1), kernel_initializer='he_uniform', kernel_constraint=self.clip_constraint)(x)

        x = layers.Conv2D(filters, (3,3), padding='same', kernel_initializer='he_uniform', kernel_constraint=self.clip_constraint)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(filters, (3,3), padding='same', kernel_initializer='he_uniform', kernel_constraint=self.clip_constraint)(x)
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
