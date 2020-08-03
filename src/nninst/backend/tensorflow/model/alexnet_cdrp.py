from typing import Dict

import numpy as np
import tensorflow as tf

from nninst import Graph
from nninst.backend.tensorflow.graph import build_graph
from nninst.utils.fs import IOAction


class AlexNetCDRP:
    """Class that defines a graph to recognize digits in the MNIST dataset."""

    def __init__(self, data_format: str = "channels_first", with_gates: bool = True):
        """Creates a model for classifying a hand-written digit.

        Args:
          data_format: Either "channels_first" or "channels_last".
            "channels_first" is typically faster on GPUs while "channels_last" is
            typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        """
        self.conv1 = tf.layers.Conv2D(
            filters=64,
            kernel_size=11,
            strides=4,
            padding="same",
            data_format=data_format,
            activation=self.gated_relu(64, "gate1"),
        )
        self.conv2 = tf.layers.Conv2D(
            filters=192,
            kernel_size=5,
            padding="same",
            data_format=data_format,
            activation=self.gated_relu(192, "gate2"),
        )
        self.conv3 = tf.layers.Conv2D(
            filters=384,
            kernel_size=3,
            padding="same",
            data_format=data_format,
            activation=self.gated_relu(384, "gate3"),
        )
        self.conv4 = tf.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            data_format=data_format,
            activation=self.gated_relu(256, "gate4"),
        )
        self.conv5 = tf.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            data_format=data_format,
            activation=self.gated_relu(256, "gate5"),
        )
        self.fc1 = tf.layers.Dense(4096, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(4096, activation=tf.nn.relu)
        # self.fc1 = tf.layers.Dense(4096, activation=self.gated_relu(4096, "gate6", is_conv=False))
        # self.fc2 = tf.layers.Dense(4096, activation=self.gated_relu(4096, "gate7", is_conv=False))
        self.fc3 = tf.layers.Dense(1000)
        self.max_pool2d = tf.layers.MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), data_format=data_format
        )
        self.dropout = tf.layers.Dropout(rate=0.5)
        self.gate_variables = {}
        self.with_gates = with_gates

    def __call__(
        self,
        inputs,
        gate_variables: Dict[str, tf.Variable] = None,
        batch_size: int = 1,
        training=False,
    ):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A logits Tensor with shape [<batch_size>, 10].
        """
        self.gate_variables = {}
        self.batch_size = batch_size
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        y = self.conv1(inputs)
        y = self.max_pool2d(y)
        y = self.conv2(y)
        y = self.max_pool2d(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.max_pool2d(y)
        y = tf.layers.flatten(y)
        y = self.dropout(y, training=training)
        y = self.fc1(y)
        y = self.dropout(y, training=training)
        y = self.fc2(y)
        y = self.fc3(y)
        if gate_variables is not None:
            gate_variables.update(self.gate_variables)
        return y

    def gated_relu(self, length, gate_name="gate", is_conv=True):
        def layer_fn(features, name=None):
            return tf.nn.relu(self.gated(features, length, gate_name, is_conv), name)

        return layer_fn

    def gated(self, input, length, name="gate", is_conv=True):
        if self.with_gates:
            return tf.multiply(
                input, tf.nn.relu(self.gate_variable(length, name, is_conv))
            )
        else:
            return input

    def gate_variable(self, length, name="gate", is_conv=True):
        # if is_conv:
        #     initial = tf.constant([1.0] * length, shape=(None, length, 1, 1))
        # else:
        #     initial = tf.constant([1.0] * length, shape=(None, length))
        v = tf.get_variable(
            name=name,
            initializer=tf.ones_initializer(),
            shape=(self.batch_size, length, 1, 1)
            if is_conv
            else (self.batch_size, length),
        )
        self.gate_variables[v.name] = v
        return v


if __name__ == "__main__":
    # with tf.Session() as sess:
    #     input = tf.placeholder(tf.float32, shape=(1, 1, 28, 28))
    #     model = LeNet()
    #     logits = model(input)
    #     graph = build_graph([input], [logits])
    #     graph.print()
    # AlexNet.graph().save()
    # graph = AlexNet.graph().load()
    # graph.print()
    pass
