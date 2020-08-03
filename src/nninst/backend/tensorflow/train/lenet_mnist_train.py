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

from nninst.backend.tensorflow.dataset import mnist
from nninst.backend.tensorflow.model import LeNet
from nninst.backend.tensorflow.utils import new_session_config
from nninst.utils.fs import abspath

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    model = LeNet(params["data_format"])
    image = features
    if isinstance(image, dict):
        image = features["image"]

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(image, training=False)
        predictions = {
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={"classify": tf.estimator.export.PredictOutput(predictions)},
        )
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            learning_rate=0.01,
            global_step=global_step * params["batch_size"],
            decay_steps=60000 * 30,
            decay_rate=0.1,
            staircase=True,
        )
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        # learning_rate = tf.train.exponential_decay(
        #     learning_rate=0.01,
        #     global_step=global_step * params["batch_size"],
        #     decay_steps=60000,
        #     decay_rate=0.95,
        #     staircase=True)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.AdamOptimizer(0.01)
        # optimizer = tf.train.AdamOptimizer(0.01, epsilon=1)

        # If we are running multi-GPU, we need to wrap the optimizer.
        if params.get("multi_gpu"):
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        # If no loss_filter_fn is passed, assume we want the default behavior,
        # which is that batch_normalization variables are excluded from loss.
        def loss_filter_fn(name):
            return "batch_normalization" not in name

        logits = model(image, training=True)
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits
        )
        weight_decay = 1e-4
        # Add weight decay to the loss.
        loss = cross_entropy + weight_decay * tf.add_n(
            [
                tf.nn.l2_loss(v)
                for v in tf.trainable_variables()
                if loss_filter_fn(v.name)
            ]
        )

        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1)
        )
        # Name the accuracy tensor "train_accuracy" to demonstrate the
        # LoggingTensorHook.
        tf.identity(accuracy[1], name="train_accuracy")
        tf.summary.scalar("train_accuracy", accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, global_step),
        )
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)
                )
            },
        )


def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == "GPU"])
    if not num_gpus:
        raise ValueError(
            "Multi-GPU mode was specified, but no GPUs "
            "were found. To use CPU, run without --multi_gpu."
        )

    remainder = batch_size % num_gpus
    if remainder:
        err = (
            "When running with multiple GPUs, batch size "
            "must be a multiple of the number of available GPUs. "
            "Found {} GPUs with a batch size of {}; try --batch_size={} instead."
        ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)


def train(
    batch_size: int = 64,
    train_epochs: int = 10,
    data_format: str = "channels_first",
    multi_gpu: bool = False,
    label: str = None,
):
    tf.logging.set_verbosity(tf.logging.INFO)
    if label is None:
        model_dir = abspath("tf/lenet/model3/")
    else:
        model_dir = abspath(f"tf/lenet/model_{label}/")
    data_dir = abspath("/home/yxqiu/data/mnist/raw")

    model_function = model_fn

    if multi_gpu:
        validate_batch_size_for_multi_gpu(batch_size)

        # There are two steps required if using multi-GPU: (1) wrap the model_fn,
        # and (2) wrap the optimizer. The first happens here, and (2) happens
        # in the model_fn itself when the optimizer is defined.
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_fn, loss_reduction=tf.losses.Reduction.MEAN
        )

    if data_format is None:
        data_format = (
            "channels_first" if tf.test.is_built_with_cuda() else "channels_last"
        )
    estimator_config = tf.estimator.RunConfig(
        keep_checkpoint_max=None, session_config=new_session_config(parallel=0)
    )
    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=model_dir,
        params={
            "data_format": data_format,
            "multi_gpu": multi_gpu,
            "batch_size": batch_size,
        },
        config=estimator_config,
    )

    for epoch in range(train_epochs):
        # Train the model
        def train_input_fn():
            # When choosing shuffle buffer sizes, larger sizes result in better
            # randomness, while smaller sizes use less memory. MNIST is a small
            # enough dataset that we can easily shuffle the full epoch.
            ds = mnist.train(data_dir)
            ds = ds.cache().shuffle(buffer_size=60000).batch(batch_size)
            return ds

        # Set up training hook that logs the training accuracy every 100 steps.
        tensors_to_log = {"train_accuracy": "train_accuracy"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100
        )
        classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

        # Evaluate the model and print results
        def eval_input_fn():
            return (
                mnist.test(data_dir)
                .batch(batch_size)
                .make_one_shot_iterator()
                .get_next()
            )

        eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(label)
    print("Evaluation results:\n\t%s" % eval_results)
    print()


if __name__ == "__main__":
    # fire.Fire(train)
    # for i in range(1):
    #     train(label=f"test{i}")
    # train(label=f"norm")
    train(label=f"dropout")
