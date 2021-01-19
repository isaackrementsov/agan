# StyleGAN2 Features written using Matchue's tutorial - https://www.youtube.com/channel/UCxBlj282mOVF2pndNPmu71w, https://github.com/manicman1999
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import layers, initializers, regularizers, constraints
import keras.backend as K

# Generator network
class Generator:

    def __init__(self, max_size, restore=False, z_length=512, upsample_size=2, depth=3):
        self.max_size = max_size
        self.z_length = z_length
        self.upsample_size = upsample_size
        self.depth = depth
        # Compute how many style blocks are needed to reach desired output size
        self.n_style_blocks = int(np.log(max_size/4)/np.log(upsample_size))

        if restore:
            self.model = self.restore()
        else:
            self.model = self.new()

    def tRGB(self, x):
        size = x.shape[2]
        scale = self.max_size//size
        vs = initializers.VarianceScaling(200/size)

        x = self.conv2d(3, (1,1), kernel_initializer=vs)(x)
        x = self.upsample((scale, scale))(x)

        return x

    def conv2d(self, filters, kernel_size=(3,3), kernel_initializer='glorot_uniform'):
        return layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer)

    def upsample(self, size):
        return layers.UpSampling2D(size=size, interpolation='bilinear')

    def block(self, x, filters, upsample=True):
        if upsample:
            y = self.upsample(self.upsample_size)(x)
        else:
            y = layers.Activation('linear')(x)

        y = layers.LeakyReLU(0.2)(y)
        y = self.conv2d(filters)(y)

        return y, self.tRGB(y)

    def new(self):
        # Latent space input
        z = keras.Input([self.z_length])

        x = layers.Dense(25*25*180, use_bias=False)(z)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((25,25,180))(x)

        x = layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(64, (5,5), strides=(1,1), padding='same', use_bias=False, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(32, (5,5), strides=(3,3), padding='same', use_bias=False, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(3, (5,5), strides=(3,3), padding='same', use_bias=False, activation='tanh')(x)

        return keras.Model(inputs=z, outputs=x)

    def restore(self):
        return keras.models.load_model('Generator', compile=False)

    def save(self):
        keras.models.save_model(self.model, 'Generator')

    def __call__(self, inputs, training=True):
        return self.model(inputs, training=training)
