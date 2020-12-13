import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from network import AGAN

aGAN = AGAN(
    noise_size=200,
    batch_size=0
)

aGAN.restore()

print('Generating sample image...')
aGAN.generate_image('test')
print('Generating sample animation...')
secs = 10
aGAN.generate_animation('test', 60*secs, secs)
