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

import fire
import tensorflow as tf

from nninst.backend.tensorflow.dataset import imagenet
from nninst.backend.tensorflow.model import ResNet50
from nninst.backend.tensorflow.utils import new_session_config
from nninst.utils.fs import abspath


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates
):
    """Get a learning rate that decays step-wise as training progresses.

    Args:
      batch_size: the number of examples processed in each training batch.
      batch_denom: this value will be used to scale the base learning rate.
        `0.1 * batch size` is divided by this number, such that when
        batch_denom == batch_size, the initial learning rate will be 0.1.
      num_images: total number of images that will be used for training.
      boundary_epochs: list of ints representing the epochs at which we
        decay the learning rate.
      decay_rates: list of floats representing the decay rates to be used
        for scaling the learning rate. It should have one more element
        than `boundary_epochs`, and all elements should have the same type.

    Returns:
      Returns a function that takes a single argument - the number of batches
      trained so far (global_step)- and returns the learning rate to be used
      for training the next batch.
    """
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    model = ResNet50()
    image = features
    if isinstance(image, dict):
        image = features["image"]

    # Generate a summary node for the images
    tf.summary.image("images", features, max_outputs=6)

    logits = model(image, mode == tf.estimator.ModeKeys.TRAIN)
    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name="cross_entropy")
    tf.summary.scalar("cross_entropy", cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def loss_filter_fn(name):
        return "batch_normalization" not in name

    weight_decay = 1e-4
    # Add weight decay to the loss.
    loss = cross_entropy + weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables() if loss_filter_fn(v.name)]
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate_fn = learning_rate_with_decay(
            batch_size=params["batch_size"],
            batch_denom=256,
            num_images=imagenet._NUM_IMAGES["train"],
            boundary_epochs=[30, 60, 80, 90],
            decay_rates=[1, 0.1, 0.01, 0.001, 1e-4],
        )
        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name="learning_rate")
        tf.summary.scalar("learning_rate", learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9
        )

        # If we are running multi-GPU, we need to wrap the optimizer.
        if params["multi_gpu"]:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(labels, predictions["classes"])
    metrics = {"accuracy": accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name="train_accuracy")
    tf.summary.scalar("train_accuracy", accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
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
    epochs_between_evals: int = 1,
    multi_gpu: bool = False,
    label: str = None,
):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    tf.logging.set_verbosity(tf.logging.INFO)
    if label is None:
        model_dir = abspath("tf/resnet-50-v2/model_train/")
    else:
        model_dir = abspath(f"tf/resnet-50-v2/model_{label}/")
    # data_dir = abspath("/home/yxqiu/data/imagenet")
    data_dir = abspath("/state/ssd0/yxqiu/data/imagenet")

    model_function = model_fn
    if multi_gpu:
        validate_batch_size_for_multi_gpu(batch_size)

        # There are two steps required if using multi-GPU: (1) wrap the model_fn,
        # and (2) wrap the optimizer. The first happens here, and (2) happens
        # in the model_fn itself when the optimizer is defined.
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_function, loss_reduction=tf.losses.Reduction.MEAN
        )

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=60 * 60,
        keep_checkpoint_max=None,
        session_config=new_session_config(parallel=0),
    )
    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=model_dir,
        config=estimator_config,
        params={"batch_size": batch_size, "multi_gpu": multi_gpu},
    )

    for epoch in range(train_epochs // epochs_between_evals):
        # Train the model
        def train_input_fn():
            return imagenet.train(
                data_dir,
                batch_size,
                num_epochs=epochs_between_evals,
                num_parallel_calls=40,
                is_shuffle=True,
                multi_gpu=multi_gpu,
            )

        # Set up training hook that logs the training accuracy every 100 steps.
        tensors_to_log = {"train_accuracy": "train_accuracy"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100
        )
        classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

        # Evaluate the model and print results
        def eval_input_fn():
            return imagenet.test(
                data_dir, batch_size, num_parallel_calls=40, multi_gpu=multi_gpu
            )

        eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(label)
    print("Evaluation results:\n\t%s" % eval_results)
    print()


if __name__ == "__main__":
    fire.Fire(train)
