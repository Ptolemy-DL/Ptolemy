#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

import os

import tensorflow as tf

from nninst import Graph
from nninst.backend.tensorflow.dataset import mnist
from nninst.backend.tensorflow.graph import MaskWeightHook
from nninst.backend.tensorflow.model import LeNet
from nninst.backend.tensorflow.trace.lenet_mnist_class_trace import (
    lenet_mnist_static_trace,
)
from nninst.utils.fs import abspath

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    model = LeNet(params["data_format"])
    image = features
    if isinstance(image, dict):
        image = features["image"]

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            evaluation_hooks=params["hooks"],
            eval_metric_ops={
                "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)
                )
            },
        )


def sparse_eval(
    graph: Graph,
    batch_size: int = 64,
    data_format: str = "channels_first",
    label: str = None,
):
    if label is None:
        model_dir = abspath("tf/lenet/model/")
    else:
        model_dir = abspath(f"tf/lenet/model_{label}/")
    data_dir = abspath("/home/yxqiu/data/mnist/raw")

    model_function = model_fn

    if data_format is None:
        data_format = (
            "channels_first" if tf.test.is_built_with_cuda() else "channels_last"
        )
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    estimator_config = tf.estimator.RunConfig(session_config=session_config)
    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=model_dir,
        params={
            "data_format": data_format,
            "batch_size": batch_size,
            "hooks": [MaskWeightHook(graph)],
        },
        config=estimator_config,
    )

    # Evaluate the model and print results
    def eval_input_fn():
        return (
            mnist.test(data_dir).batch(batch_size).make_one_shot_iterator().get_next()
        )

    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(label)
    print("Evaluation results:\n\t%s" % eval_results)
    print()


if __name__ == "__main__":
    # fire.Fire(train)
    # eval(label="import")
    threshold = 0.5
    label = "early"
    # trace = lenet_mnist_trace(threshold, label).load()
    trace = lenet_mnist_static_trace(threshold, label).load()
    graph = LeNet.graph().load()
    graph.load_attrs(trace)
    sparse_eval(graph=graph)
