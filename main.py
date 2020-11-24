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

aGAN = AGAN(100, BATCH_SIZE)
aGAN.train(training_data, 4000, 5, 0)
