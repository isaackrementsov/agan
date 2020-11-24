import tensorflow as tf
import numpy as np

def to_array(path):
    img = tf.io.read_file('assets/' + path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return (img - 0.5)*2

def split(batch_size, array):
    split_array = []
    section = []
    i = 0

    for elem in array:
        i += 1
        section.append(elem)

        if i == batch_size:
            split_array.append(section)
            section = []
            i = 0

    return tf.convert_to_tensor(split_array)
