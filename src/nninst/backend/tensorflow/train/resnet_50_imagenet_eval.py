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

from nninst.backend.tensorflow.dataset import imagenet
from nninst.backend.tensorflow.model import ResNet50
from nninst.backend.tensorflow.utils import new_session_config
from nninst.utils.fs import abspath

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def model_fn(features, labels, mode):
    """The model_fn argument for creating an Estimator."""
    model = ResNet50()
    image = features
    if isinstance(image, dict):
        image = features["image"]

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

        # cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        #
        # def loss_filter_fn(name):
        #     return 'batch_normalization' not in name
        #
        # weight_decay=1e-4
        # # Add weight decay to the loss.
        # loss = cross_entropy + weight_decay * tf.add_n(
        #     [tf.nn.l2_loss(v) for v in tf.trainable_variables()
        #      if loss_filter_fn(v.name)])
        #
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)
                )
            },
        )


def eval(batch_size: int = 64, label: str = None):
    # tf.logging.set_verbosity(tf.logging.INFO)
    if label is None:
        model_dir = abspath("tf/resnet-50-v2/model/")
    else:
        model_dir = abspath(f"tf/resnet-50-v2/model_{label}/")
    # data_dir = abspath("/home/yxqiu/data/imagenet")
    data_dir = abspath("/state/ssd0/yxqiu/data/imagenet")

    model_function = model_fn

    estimator_config = tf.estimator.RunConfig(
        session_config=new_session_config(parallel=0)
    )
    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=model_dir, config=estimator_config
    )

    # Evaluate the model and print results
    def eval_input_fn():
        return imagenet.test(data_dir, batch_size, num_parallel_calls=40)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(label)
    print("Evaluation results:\n\t%s" % eval_results)
    print()


if __name__ == "__main__":
    # fire.Fire(train)
    # eval(label="import")
    eval(label="train_start")
