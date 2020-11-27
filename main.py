import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras

from network import AGAN
from utils import to_array, split

BATCH_SIZE = 90

train_images = [to_array(img) for img in os.listdir('assets/')]
training_data = split(BATCH_SIZE, train_images)
print('Done preparing training data')

aGAN = AGAN(
    noise_size=100,
    batch_size=BATCH_SIZE
)
aGAN.restore()

try:
    aGAN.train(
        dataset=training_data,
        epochs=4000,
        example_interval=5,
        save_interval=50,
        example_offset=6815
    )
except KeyboardInterrupt:
    try:
        print('Saving...')
        aGAN.save()
    except KeyboardInterrupt:
        print('Failed to save model because the program was force-stopped')
    finally:
        quit()
