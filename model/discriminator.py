import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Discriminator network
class Discriminator:

    def __init__(self, max_size, depth=3, restore=False, downsample_size=2):
        # Input image resolution
        self.max_size = max_size
        # Determines number of filters per convolution (first layer depth)
        self.depth = depth
        # Factor by which to downsample between blocks
        self.downsample_size = downsample_size

        # Either restore or create a new model
        if restore:
            self.model = self.restore()
        else:
            self.model = self.new()

    # Block of discriminator layers
    def block(self, x, filters, avg_pooling=True):
        # Residual output
        r = layers.Conv2D(filters, (1,1), kernel_initializer='he_uniform')(x)

        # Apply two sets of convolutions
        x = layers.Conv2D(filters, (3,3), padding='same', kernel_initializer='he_uniform')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(filters, (3,3), padding='same', kernel_initializer='he_uniform')(x)
        x = layers.LeakyReLU(0.2)(x)
        # Add the residual output
        x = layers.add([x,r])

        # Downsize the input
        if avg_pooling:
            x = layers.AveragePooling2D(self.downsample_size)(x)

        return x

    def new(self):
        # H/W of last feature map before dense layers
        min_size = 4
        # Number of downsizing blocks required
        n_layers = int(np.log(min_size/self.max_size)/np.log(self.downsample_size))

        # Input RGB image
        x = keras.Input([self.max_size, self.max_size, 3])
        y = x
        # Depth parameter determines initial number of filters
        filters = self.depth

        # Add the number of blocks required to go from input res => 4x4
        for i in range(n_layers):
            filters *= 2
            y = self.block(y, filters)

        # Final convolution before dense layer
        y = self.block(y, filters, avg_pooling=False)
        # Flatten final convolution feature map and output a score
        y = layers.Flatten()(y)
        y = layers.Dense(1, kernel_initializer='he_uniform')(y)

        return keras.Model(inputs=x, outputs=y)

    def __call__(self, inputs):
        return self.model(inputs)

    # Load saved model
    def restore(self):
        return keras.models.load_model('Discriminator', compile=False)

    # Save model for loading later
    def save(self):
        return keras.models.save_model(self.model, 'Discriminator')
