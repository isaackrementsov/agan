import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading

from network import AGAN
from utils import to_array, gen_batches

BATCH_SIZE = 25

paths = os.listdir('assets/')
training_data = gen_batches(paths, BATCH_SIZE)
print('Done preparing training data')

aGAN = AGAN(
    noise_size=200,
    batch_size=BATCH_SIZE
)
aGAN.restore()

try:
    aGAN.train(
        dataset=training_data,
        epochs=12000,
        example_interval=5,
        save_interval=100
    )
except KeyboardInterrupt:
    try:
        print('Saving...')
        aGAN.save()

        print('Clearing session...')
        keras.backend.clear_session()
    except KeyboardInterrupt:
        print('Failed to save model because the program was force-stopped')
    finally:
        quit()
