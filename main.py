import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

from network import AGAN

BATCH_SIZE = 256

(train_images, labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images - 127.5) / 127.5
training_data = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(BATCH_SIZE)

aGAN = AGAN(100, BATCH_SIZE)
aGAN.train(training_data, 50, 100, 0)
