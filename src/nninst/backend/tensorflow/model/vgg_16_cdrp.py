from functools import partial
from typing import Dict

import tensorflow as tf
from tensorflow.contrib import layers, slim
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16 as slim_vgg_16
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_arg_scope
from tensorflow.python.ops import array_ops, variable_scope

from nninst import Graph
from nninst.backend.tensorflow.graph import build_graph
from nninst.utils.fs import IOAction


class VGG16CDRP:
    def __init__(self, with_gates: bool = True):
        self.gate_variables = {}
        self.with_gates = with_gates

    def __call__(
        self,
        inputs,
        gate_variables: Dict[str, tf.Variable] = None,
        batch_size: int = 1,
        training=False,
    ):
        self.gate_variables = {}
        with slim.arg_scope(vgg_arg_scope()):
            output = self.vgg_16_cdrp(inputs, is_training=training)[0]
            if gate_variables is not None:
                gate_variables.update(self.gate_variables)
            return output

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
        if is_conv:
            initial = tf.constant([1.0] * length, shape=(1, length))
        else:
            initial = tf.constant([1.0] * length, shape=(1, length))
        v = tf.get_variable(name=name, initializer=initial)
        self.gate_variables[v.name] = v
        return v

    def vgg_16_cdrp(
        self,
        inputs,
        num_classes=1000,
        is_training=True,
        dropout_keep_prob=0.5,
        spatial_squeeze=True,
        scope="vgg_16",
    ):
        """Oxford Net VGG 16-Layers version D Example.

        Note: All the fully_connected layers have been transformed to conv2d layers.
              To use in classification mode, resize input to 224x224.

        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes.
          is_training: whether or not the model is being trained.
          dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
          spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
          scope: Optional scope for the variables.

        Returns:
          the last op containing the log predictions and end_points dict.
        """
        self.gate_count = 0
        with variable_scope.variable_scope(scope, "vgg_16", [inputs]) as sc:
            end_points_collection = sc.original_name_scope + "_end_points"
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with arg_scope(
                [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
                outputs_collections=end_points_collection,
            ):

                def gated_conv2d(input, num_outputs, *args, **kwargs):
                    self.gate_count += 1
                    return layers.conv2d(
                        input,
                        num_outputs,
                        activation_fn=self.gated_relu(
                            num_outputs, f"gate{self.gate_count}"
                        ),
                        *args,
                        **kwargs,
                    )

                net = layers_lib.repeat(
                    inputs, 2, gated_conv2d, 64, [3, 3], scope="conv1"
                )
                net = layers_lib.max_pool2d(net, [2, 2], scope="pool1")
                net = layers_lib.repeat(
                    net, 2, gated_conv2d, 128, [3, 3], scope="conv2"
                )
                net = layers_lib.max_pool2d(net, [2, 2], scope="pool2")
                net = layers_lib.repeat(
                    net, 3, gated_conv2d, 256, [3, 3], scope="conv3"
                )
                net = layers_lib.max_pool2d(net, [2, 2], scope="pool3")
                net = layers_lib.repeat(
                    net, 3, gated_conv2d, 512, [3, 3], scope="conv4"
                )
                net = layers_lib.max_pool2d(net, [2, 2], scope="pool4")
                net = layers_lib.repeat(
                    net, 3, gated_conv2d, 512, [3, 3], scope="conv5"
                )
                net = layers_lib.max_pool2d(net, [2, 2], scope="pool5")
                # Use conv2d instead of fully_connected layers.
                net = layers.conv2d(net, 4096, [7, 7], padding="VALID", scope="fc6")
                net = layers_lib.dropout(
                    net, dropout_keep_prob, is_training=is_training, scope="dropout6"
                )
                net = layers.conv2d(net, 4096, [1, 1], scope="fc7")
                net = layers_lib.dropout(
                    net, dropout_keep_prob, is_training=is_training, scope="dropout7"
                )
                net = layers.conv2d(
                    net,
                    num_classes,
                    [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope="fc8",
                )
                # Convert end_points_collection into a end_point dict.
                end_points = utils.convert_collection_to_dict(end_points_collection)
                if spatial_squeeze:
                    net = array_ops.squeeze(net, [1, 2], name="fc8/squeezed")
                    end_points[sc.name + "/fc8"] = net
                return net, end_points

    vgg_16_cdrp.default_image_size = 224


if __name__ == "__main__":
    # with tf.Session() as sess:
    #     input = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    #     model = VGG16()
    #     logits = model(input, training=False)
    #     summary_writer = SummaryWriter(abspath("tmp/logs"), sess.graph)
    #     graph = build_graph([input], [logits])
    #     graph.print()
    # graph = InceptionV4.create_graph()
    # print()
    # VGG16.graph().save()
    pass
