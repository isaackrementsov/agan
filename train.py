import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading

from model.network import AGAN
from utils import to_array, gen_batches

BATCH_SIZE = 100

paths = os.listdir('assets/')
training_data = gen_batches(paths, BATCH_SIZE)
print('Done preparing training data')

aGAN = AGAN(
    batch_size=BATCH_SIZE,
    resolution=256
)
aGAN.new()

try:
    aGAN.train(
        dataset=training_data,
        epochs=12000,
        example_interval=5,
        save_interval=100
    )
except Exception as e:
    print(e)
    print('Training paused')
finally:
    print('Saving...')
    aGAN.save()

    print('Clearing session...')
    keras.backend.clear_session()

    quit()
