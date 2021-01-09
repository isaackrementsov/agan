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
        return K.clip(weights, self.clip_value_min, self.clip_value_max)

    def get_config(self):
        return {'clip_value_max': self.clip_value_max, 'clip_value_min': self.clip_value_min}

# Discriminator network
class Discriminator:

    def __init__(self, max_size, depth=3, restore=False, downsample_size=2, clip_range=(0.01,-0.01)):
        # Input image resolution
        self.max_size = max_size
        # Determines number of filters per convolution (first layer depth)
        self.depth = depth
        # Factor by which to downsample between blocks
        self.downsample_size = downsample_size
        # Weight clipping constraint
        self.clip_constraint = Clip(*clip_range)

        # Either restore or create a new model
        if restore:
            self.model = self.restore()
        else:
            self.model = self.new()

    # Block of discriminator layers
    def block(self, x, filters, avg_pooling=True):
        # Residual output
        r = layers.Conv2D(filters, (1,1), kernel_constraint=self.clip_constraint)(x)

        # Apply two sets of convolutions
        x = layers.Conv2D(filters, (3,3), padding='same', kernel_constraint=self.clip_constraint)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(filters, (3,3), padding='same', kernel_constraint=self.clip_constraint)(x)
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
        y = layers.Dense(1)(y)

        return keras.Model(inputs=x, outputs=y)

    def __call__(self, inputs):
        return self.model(inputs)

    # Load saved model
    def restore(self):
        return keras.models.load_model('Discriminator', compile=False)

    # Save model for loading later
    def save(self):
        return keras.models.save_model(self.model, 'Discriminator')
