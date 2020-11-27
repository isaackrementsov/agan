import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from network import AGAN

aGAN = AGAN(
    noise_size=100,
    batch_size=0
)
aGAN.restore(compile=False)
aGAN.generate_image('test')
#aGAN.generate_animation('test', 60*5)
