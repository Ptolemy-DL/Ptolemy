import tensorflow as tf

from nninst import Graph
from nninst.backend.tensorflow.graph import build_graph
from nninst.utils.fs import IOAction


class AlexNet:
    """Class that defines a graph to recognize digits in the MNIST dataset."""

    def __init__(self, data_format: str = "channels_first"):
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
            activation=tf.nn.relu,
        )
        self.conv2 = tf.layers.Conv2D(
            filters=192,
            kernel_size=5,
            padding="same",
            data_format=data_format,
            activation=tf.nn.relu,
        )
        self.conv3 = tf.layers.Conv2D(
            filters=384,
            kernel_size=3,
            padding="same",
            data_format=data_format,
            activation=tf.nn.relu,
        )
        self.conv4 = tf.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            data_format=data_format,
            activation=tf.nn.relu,
        )
        self.conv5 = tf.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            data_format=data_format,
            activation=tf.nn.relu,
        )
        self.fc1 = tf.layers.Dense(4096, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(4096, activation=tf.nn.relu)
        self.fc3 = tf.layers.Dense(1000)
        self.max_pool2d = tf.layers.MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), data_format=data_format
        )
        self.dropout = tf.layers.Dropout(rate=0.5)

    def __call__(self, inputs, training=False):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A logits Tensor with shape [<batch_size>, 10].
        """
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        y = self.conv1(inputs)
        y = self.max_pool2d(y)
        y = self.conv2(y)
        y = self.max_pool2d(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.max_pool2d(y)
        y = tf.layers.Flatten()(y)
        y = self.dropout(y, training=training)
        y = self.fc1(y)
        y = self.dropout(y, training=training)
        y = self.fc2(y)
        return self.fc3(y)

    @classmethod
    def create_graph(cls, input_name: str = "IteratorGetNext:0") -> Graph:
        with tf.Session().as_default() as sess:
            input = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
            new_graph = build_graph([input], [cls()(input)])
            new_graph.rename(new_graph.id(input.name), input_name)
            sess.close()
            return new_graph

    @classmethod
    def graph(cls) -> IOAction[Graph]:
        path = "store/graph/alexnet.pkl"
        return IOAction(path, init_fn=lambda: AlexNet.create_graph())


if __name__ == "__main__":
    AlexNet.graph().save()
    # graph = AlexNet.graph().load()
    # graph.print()
