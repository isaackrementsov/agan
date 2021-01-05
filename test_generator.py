import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from model.network import AGAN

aGAN = AGAN(
    batch_size=10,
    resolution=256
)

aGAN.restore()

print('Generating sample image...')
aGAN.generate_image('test')
print('Generating sample animation...')
secs = 2
aGAN.generate_animation('test', 60*secs, secs)
