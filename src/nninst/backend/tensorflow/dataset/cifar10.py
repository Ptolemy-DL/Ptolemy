import numpy as np
import tensorflow as tf
from keras.datasets import cifar10


def normalize(x):
    mean = 120.707
    std = 64.15
    return (x - mean) / (std + 1e-7)


def dataset(class_id, image_id, normed=True):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    filter = y_train[:, 0] == class_id
    x_train = x_train[filter][image_id]
    y_train = y_train[filter][image_id]
    x_train = np.expand_dims(x_train.astype(np.float32), axis=0)
    dataset = tf.data.Dataset.from_tensors(
        (normalize(x_train) if normed else x_train, y_train)
    )
    return dataset


def train(directory, class_id, image_id, normed=True):
    """tf.data.Dataset object for MNIST training data."""
    return dataset(class_id, image_id, normed)


def test(directory, class_id, image_id, normed=True):
    """tf.data.Dataset object for MNIST test data."""
    return dataset(class_id, image_id, normed)
