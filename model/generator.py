# StyleGAN2 Features written using Matchue's tutorial - https://www.youtube.com/channel/UCxBlj282mOVF2pndNPmu71w, https://github.com/manicman1999
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import layers, initializers, regularizers, constraints
import keras.backend as K


# Custom mod-demod convolutional layer
class ModulatedConv2D(layers.Layer):

    def __init__(self, filters, kernel_size,
                strides=1, padding='valid', dilation_rate=(1,1),
                kernel_initializer='glorot_uniform', kernel_regularizer=None,
                activity_regularizer=None, kernel_constraint=None,
                demod=True, **kwargs):

        super(ModulatedConv2D, self).__init__(**kwargs)

        self.filters = filters
        self.rank = 2
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.demod = demod
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=2)]
        self.epsilon = 1e-8

    def build(self, input_shape):
        channel_dim = input_shape[0][-1]

        if channel_dim is None:
            raise ValueError('The channel dimension of the input shape should be defined. Found `None`.')

        # Kernel shape is (3,3,input_maps,output_maps)
        kernel_shape = self.kernel_size + (channel_dim, self.filters)

        if input_shape[1][-1] != channel_dim:
            raise ValueError('The last dimension of the style input should be equal to the channel dimension')

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.built = True

    def call(self, inputs):
        x = inputs[0]
        # Expand style vector to shape (batch, 1, 1, w_length, 1)
        w = K.expand_dims(K.expand_dims(K.expand_dims(inputs[1], 1), 1), -1)

        # Add batch dimension to initial kernel weights
        weights = K.expand_dims(self.kernel, axis = 0)
        # Modulate kernel weights
        weights *= (w + 1)

        # Demodulate kernel weights
        if self.demod:
            # Get L2 norm of output weights (and prevent division by zero)
            norm = K.sqrt(K.sum(K.square(weights), axis=[1,2,3], keepdims=True) + self.epsilon)
            # Normalize weights
            weights /= norm

        x_shape = K.shape(x)
        # Fuse input batches into channel dimension
        x = K.reshape(x, [1, x_shape[1], x_shape[2], -1])
        # Fuse kernels to be in a single layer weight-style instance
        weights = tf.reshape(tf.transpose(weights, [1,2,3,0,4]), [weights.shape[1], weights.shape[2], weights.shape[3], -1])

        # Perform 3x3 convolution with styled kernels
        x = tf.nn.conv2d(x, weights, strides=self.strides, padding='SAME', data_format='NHWC')

        # Separate output channels back into batches
        x_shape = K.shape(x)
        x = K.reshape(x, [-1, x_shape[1], x_shape[2], self.filters])

        return x

    def compute_output_shape(self, input_shape):
        input_space = input_shape[0][1:-1]
        output_space = []

        for i in range(len(input_space)):
            output_dim = conv_utils.conv_output_length(
                input_space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i]
            )

            output_space.append(output_dim)

        # Output shape = (batch_size, *output_space, filters)
        return (input_shape[0],) + tuple(output_space) + (self.filters,)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_contraint': constraints.serialize(self.kernel_constraint),
            'demod': self.demod
        }

        base_config = super(ModulatedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Style mapping network
class StyleMapper:

    def __init__(self, restore=False, w_length=512, latent_size=512):
        self.w_length = w_length
        self.latent_size = latent_size

        # Make new or restore model
        if restore:
            self.model = self.restore()
        else:
            self.model = self.new()

    # Load saved model
    def restore(self):
        return keras.models.load_model('StyleMapper', compile=False)

    # Save current model state
    def save(self):
        self.model.save('StyleMapper')

    # Create new model
    def new(self):
        # Make network from 8 fully connected layers

        model = keras.Sequential()
        # Transforms point in latent space to style vector
        model.add(layers.Dense(self.w_length, input_shape=[self.latent_size]))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dense(self.w_length))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dense(self.w_length))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dense(self.w_length))
        model.add(layers.LeakyReLU(0.2))

        return model

    def __call__(self, inputs, training=True):
        return self.model(inputs, training=training)

# Generator ("synthesis") network
class Generator:

    def __init__(self, max_size, batch_size, restore=False, w_length=512, upsample_size=2, depth=3):
        self.max_size = max_size
        self.w_length = w_length
        self.upsample_size = upsample_size
        self.depth = depth
        self.seed = tf.ones([batch_size, w_length])

        if restore:
            self.model = self.restore()
        else:
            self.model = self.new()

    def tRGB(self, x, w):
        size = x.shape[2]
        scale = self.max_size//size
        vs = initializers.VarianceScaling(200/size)

        x = self.mod_conv2d(3, (1,1), kernel_initializer=vs, demod=False)([x,w])
        x = self.upsample((scale, scale))(x)

        return x

    def crop(self, inputs):
        h = inputs[1].shape[1]
        w = inputs[1].shape[2]

        return inputs[0][:, :h, :w, :]

    def mod_conv2d(self, filters, kernel_size=(3,3), kernel_initializer='he_uniform', demod=True):
        return ModulatedConv2D(filters, kernel_size, kernel_initializer=kernel_initializer, demod=demod)

    def upsample(self, size):
        return layers.UpSampling2D(size=size, interpolation='bilinear')

    def block(self, x, iw, ib, filters, upsample=True):
        if upsample:
            y = self.upsample(self.upsample_size)(x)
        else:
            y = layers.Activation('linear')(x)

        r = lambda: np.random.normal()
        suffix = str(y.shape[1]) + str(r() + r() + r())
        if upsample:
            suffix = 'upsample' + suffix

        # Style vector for use in tRGB
        w_rgb = layers.Dense(filters, kernel_initializer=initializers.VarianceScaling(200/y.shape[2]), name='dense_wrgb' + suffix)(iw)
        # Reshape style vector
        w = layers.Dense(x.shape[-1], kernel_initializer='he_uniform', name='dense_w' + suffix)(iw)
        # Crop noise to fit image
        b = layers.Lambda(self.crop)([ib, y])
        # Pass noise through a dense layer to fit filter size
        d = layers.Dense(filters, kernel_initializer='zeros', name='dense_d' + suffix)(b)

        y = self.mod_conv2d(filters)([y, w])
        y = layers.add([y, d])
        y = layers.LeakyReLU(0.2)(y)

        w = layers.Dense(filters, kernel_initializer='he_uniform')(iw)
        d = layers.Dense(filters, kernel_initializer='zeros')(b)

        y = self.mod_conv2d(filters)([y, w])
        y = layers.add([y, d])
        y = layers.LeakyReLU(0.2)(y)

        return y, self.tRGB(y, w_rgb)

    def new(self):
        # Compute how many style blocks are needed to reach desired output size
        self.n_style_blocks = int(np.log(self.max_size/4)/np.log(self.upsample_size))

        # Style vectors
        w = [keras.Input([self.w_length]) for i in range(self.n_style_blocks + 1)]
        # Constant "seed" vector for generator
        c = keras.Input([self.w_length])
        # Random noise vectors
        b = keras.Input([self.max_size,self.max_size,1])

        x = layers.Dense(self.depth*4**3, activation='relu', kernel_initializer='normal')(c)
        x = layers.Reshape([4, 4, 4*self.depth])(x)

        # Stores outputs from each block for progressive growth-like training
        res = []

        # Initial filter factor so that last block has "self.depth" filters
        filters = int(2**self.n_style_blocks*self.depth)
        # Initial 4x4 convolution layer
        x, r = self.block(x, w[0], b, filters, upsample=False)

        for i in range(self.n_style_blocks):
            filters //= 2
            x, r = self.block(x, w[i + 1], b, filters)
            res.append(r)

        x = layers.add(res)

        return keras.Model(inputs=[w, b, c], outputs=x)

    def restore(self):
        return keras.models.load_model('Generator', compile=False)

    def save(self):
        keras.models.save_model(self.model, 'Generator')

    def __call__(self, inputs, training=True):
        w = inputs[0]
        b = inputs[1]

        return self.model([w, b, self.seed], training=training)
