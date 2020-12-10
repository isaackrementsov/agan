from keras.layers import Layer
import keras.backend as K

class AdaIN(Layer):

    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True **kwargs):
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        dim = input_shape[0][self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of ' + 'input tensor should have defined dimension but the layer recieved an input recieved an input with shape ' + str(input_shape[0]))

        super(AdaIN, self).build(input_shape)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))

        beta = inputs[1]
        gamma = inputs[2]

        if self.axis is not None:
            del reduction_axes[self.axis]

        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean)/stddev

        return normed*gamma + beta

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }

        base_config = super(SPADE, self).get_config()

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class SPADE(Layer):
    def __init__(self,
             axis=-1,
             momentum=0.99,
             epsilon=1e-3,
             center=True,
             scale=True,
             **kwargs):
        super(SPADE, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale


    def build(self, input_shape):

        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')

        super(SPADE, self).build(input_shape)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])

        beta = inputs[1]
        gamma = inputs[2]

        reduction_axes = [0, 1, 2]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(SPADE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):

        return input_shape[0]    
