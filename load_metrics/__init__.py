import tensorflow as tf
from compute import average, disorder, potential_energy, kinetic_energy

def batch(func, array):
    return [func(elem) for elem in array]

def summarize(image_batch):
    return tf.constant([
        batch(average, image_batch),
        batch(disorder, image_batch),
        batch(kinetic_energy, image_batch),
        batch(potential_energy, image_batch)
    ])
