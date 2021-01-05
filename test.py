import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from model.network import AGAN
from mapper_model.network import LatentMapper

# Restore both networks
mapper = StyleMapper(0, 0, 0, lambda x: 0)
mapper.restore()

aGAN = AGAN(0, 256)
aGAN.restore()

# Number of values used in physical summary vector
VEC_SIZE = 4
# The number of style blocks in the generator (and the maximum size of the style vector)
n_style_blocks = aGAN.n_blocks()

# Random "physical" summary vector data
sample_data = tf.random.uniform([n_blocks,VEC_SIZE], minval=-10, maxval=10)
# Data mapped to points
latent_points = mapper.Z(sample_data)

print('Generating sample image...')
aGAN.generate_from_mapper('test', latent_points)
