import numpy as np
import tensorflow as tf
from tensorpack import (
    AvgPooling,
    BatchNorm,
    Conv2D,
    FullyConnected,
    GlobalAvgPooling,
    TowerContext,
    dataset,
)

from nninst import Graph
from nninst.backend.tensorflow.graph import build_graph
from nninst.utils.fs import IOAction

ds = dataset.Cifar10("train")
pp_mean = ds.get_per_pixel_mean()


def normalize_cifar_densenet(image):
    image = image.astype(np.float32)
    return (image - pp_mean) / 128.0 - 1


def normalize_cifar_densenet_with_grad(image):
    image = image.astype(np.float32)
    image = (image - pp_mean) / 128.0 - 1

    def grad(dmdp):
        return dmdp / 128.0

    return image, grad


class DenseNet:
    def __init__(self, depth=40):
        self.N = int((depth - 4) / 3)
        self.growthRate = 12

    def __call__(self, inputs, training=False):
        def conv(name, layer, channel, stride):
            return Conv2D(
                name,
                layer,
                channel,
                3,
                stride=stride,
                nl=tf.identity,
                use_bias=False,
                W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / channel)),
            )

        def add_layer(name, layer):
            shape = layer.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                c = BatchNorm("bn1", layer)
                c = tf.nn.relu(c)
                c = conv("conv1", c, self.growthRate, 1)
                layer = tf.concat([c, layer], 3)
            return layer

        def add_transition(name, layer):
            shape = layer.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                layer = BatchNorm("bn1", layer)
                layer = tf.nn.relu(layer)
                layer = Conv2D(
                    "conv1",
                    layer,
                    in_channel,
                    1,
                    stride=1,
                    use_bias=False,
                    nl=tf.nn.relu,
                )
                layer = AvgPooling("pool", layer, 2)
            return layer

        def densenet(inputs):
            layer = conv("conv0", inputs, 16, 1)
            with tf.variable_scope("block1") as scope:

                for i in range(self.N):
                    layer = add_layer("dense_layer.{}".format(i), layer)
                layer = add_transition("transition1", layer)

            with tf.variable_scope("block2") as scope:

                for i in range(self.N):
                    layer = add_layer("dense_layer.{}".format(i), layer)
                layer = add_transition("transition2", layer)

            with tf.variable_scope("block3") as scope:

                for i in range(self.N):
                    layer = add_layer("dense_layer.{}".format(i), layer)
            layer = BatchNorm("bnlast", layer)
            layer = tf.nn.relu(layer)
            layer = GlobalAvgPooling("gap", layer)
            return FullyConnected("linear", layer, out_dim=10, nl=tf.identity)

        with TowerContext("", is_training=training):
            logits = densenet(inputs)

        return logits

    @classmethod
    def create_graph(cls, input_name: str = "IteratorGetNext:0") -> Graph:
        with tf.Session().as_default() as sess:
            input = tf.placeholder(tf.float32, shape=(1, 32, 32, 3))
            new_graph = build_graph([input], [cls()(input)])
            new_graph.rename(new_graph.id(input.name), input_name)
            sess.close()
            for op in new_graph.ops:
                print(op)
            return new_graph

    @classmethod
    def graph(cls) -> IOAction[Graph]:
        path = "store/graph/densenet.pkl"
        return IOAction(path, init_fn=lambda: DenseNet.create_graph())


if __name__ == "__main__":
    DenseNet.graph().save()
    graph = DenseNet.graph().load()
    graph.print()
