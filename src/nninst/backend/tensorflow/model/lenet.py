import tensorflow as tf

from nninst import Graph
from nninst.backend.tensorflow.graph import build_graph
from nninst.utils.fs import IOAction


class LeNet:
    """Class that defines a graph to recognize digits in the MNIST dataset."""

    def __init__(self, data_format: str = "channels_first"):
        """Creates a model for classifying a hand-written digit.

        Args:
          data_format: Either "channels_first" or "channels_last".
            "channels_first" is typically faster on GPUs while "channels_last" is
            typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        """
        # if data_format == "channels_first":
        #     self._input_shape = [-1, 1, 28, 28]
        # else:
        #     assert data_format == "channels_last"
        #     self._input_shape = [-1, 28, 28, 1]

        self.conv1 = tf.layers.Conv2D(
            6, 5, data_format=data_format, activation=tf.nn.relu
        )
        self.conv2 = tf.layers.Conv2D(
            16, 5, data_format=data_format, activation=tf.nn.relu
        )
        self.fc1 = tf.layers.Dense(120, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(84, activation=tf.nn.relu)
        self.fc3 = tf.layers.Dense(10)
        self.max_pool2d = tf.layers.MaxPooling2D(
            (2, 2), (2, 2), padding="same", data_format=data_format
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
        # y = tf.reshape(inputs, self._input_shape)
        # y = self.conv1(y)
        y = self.conv1(inputs)
        y = self.max_pool2d(y)
        y = self.conv2(y)
        y = self.max_pool2d(y)
        y = tf.layers.flatten(y)
        y = self.fc1(y)
        y = self.dropout(y, training=training)
        y = self.fc2(y)
        y = self.dropout(y, training=training)
        return self.fc3(y)

    @classmethod
    def create_graph(cls, input_name: str = "IteratorGetNext:0") -> Graph:
        input = tf.placeholder(tf.float32, shape=(1, 1, 28, 28))
        new_graph = build_graph([input], [cls()(input)])
        new_graph.rename(new_graph.id(input.name), input_name)
        return new_graph

    @classmethod
    def graph(cls) -> IOAction[Graph]:
        path = "store/graph/lenet.pkl"
        return IOAction(path, init_fn=lambda: LeNet.create_graph())


if __name__ == "__main__":
    # with tf.Session() as sess:
    #     input = tf.placeholder(tf.float32, shape=(1, 1, 28, 28))
    #     model = LeNet()
    #     logits = model(input)
    #     graph = build_graph([input], [logits])
    #     graph.print()
    LeNet.graph().save()
