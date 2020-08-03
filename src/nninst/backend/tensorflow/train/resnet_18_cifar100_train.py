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
from tensorflow.contrib.data.python.ops import threadpool

from nninst.backend.tensorflow.dataset import imagenet, imagenet_preprocessing
from nninst.backend.tensorflow.dataset.cifar100_main import input_fn
from nninst.backend.tensorflow.model import ResNet50
from nninst.backend.tensorflow.model.resnet_18_cifar100 import ResNet18Cifar100
from nninst.backend.tensorflow.utils import new_session_config
from nninst.utils.fs import abspath

_NUM_IMAGES = {"train": 50000, "validation": 10000}


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size,
    batch_denom,
    num_images,
    boundary_epochs,
    decay_rates,
    base_lr=0.1,
    warmup=False,
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
      base_lr: Initial learning rate scaled based on batch_denom.
      warmup: Run a 5 epoch warmup to the initial lr.
    Returns:
      Returns a function that takes a single argument - the number of batches
      trained so far (global_step)- and returns the learning rate to be used
      for training the next batch.
    """
    initial_learning_rate = base_lr * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Reduce the learning rate at certain epochs.
    # CIFAR-10: divide by 10 at epoch 100, 150, and 200
    # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        """Builds scaled learning rate function with 5 epoch warm up."""
        lr = tf.train.piecewise_constant(global_step, boundaries, vals)
        if warmup:
            warmup_steps = int(batches_per_epoch * 5)
            warmup_lr = (
                initial_learning_rate
                * tf.cast(global_step, tf.float32)
                / tf.cast(warmup_steps, tf.float32)
            )
            return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
        return lr

    return learning_rate_fn


def resnet_model_fn(
    features,
    labels,
    mode,
    model_class,
    weight_decay,
    learning_rate_fn,
    momentum,
    loss_scale,
    loss_filter_fn=None,
    dtype=tf.float32,
    fine_tune=False,
):
    """Shared functionality for different resnet model_fns.
    Initializes the ResnetModel representing the model layers
    and uses that model to build the necessary EstimatorSpecs for
    the `mode` in question. For training, this means building losses,
    the optimizer, and the train op that get passed into the EstimatorSpec.
    For evaluation and prediction, the EstimatorSpec is returned without
    a train op, but with the necessary parameters for the given mode.
    Args:
      features: tensor representing input images
      labels: tensor representing class labels for all input images
      mode: current estimator mode; should be one of
        `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
      model_class: a class representing a TensorFlow model that has a __call__
        function. We assume here that this is a subclass of ResnetModel.
      resnet_size: A single integer for the size of the ResNet model.
      weight_decay: weight decay loss rate used to regularize learned variables.
      learning_rate_fn: function that returns the current learning rate given
        the current global_step
      momentum: momentum term used for optimization
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      resnet_version: Integer representing which version of the ResNet network to
        use. See README for details. Valid values: [1, 2]
      loss_scale: The factor to scale the loss for numerical stability. A detailed
        summary is present in the arg parser help text.
      loss_filter_fn: function that takes a string variable name and returns
        True if the var should be included in loss calculation, and False
        otherwise. If None, batch_normalization variables will be excluded
        from the loss.
      dtype: the TensorFlow dtype to use for calculations.
      fine_tune: If True only train the dense layers(final layers).
    Returns:
      EstimatorSpec parameterized according to the input params and the
      current mode.
    """

    # Generate a summary node for the images
    tf.summary.image("images", features, max_outputs=6)
    # Checks that features/images have same data type being used for calculations.
    assert features.dtype == dtype

    model = model_class()

    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    # This acts as a no-op if the logits are already in fp32 (provided logits are
    # not a SparseTensor). If dtype is is low precision, logits must be cast to
    # fp32 for numerical stability.
    logits = tf.cast(logits, tf.float32)

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={"predict": tf.estimator.export.PredictOutput(predictions)},
        )

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name="cross_entropy")
    tf.summary.scalar("cross_entropy", cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return "batch_normalization" not in name

    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [
            tf.nn.l2_loss(tf.cast(v, tf.float32))
            for v in tf.trainable_variables()
            if loss_filter_fn(v.name)
        ]
    )
    tf.summary.scalar("l2_loss", l2_loss)
    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name="learning_rate")
        tf.summary.scalar("learning_rate", learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum
        )

        def _dense_grad_filter(gvs):
            """Only apply gradient updates to the final layer.
            This function is used for fine tuning.
            Args:
              gvs: list of tuples with gradients and variable info
            Returns:
              filtered gradients so that only the dense layer remains
            """
            return [(g, v) for g, v in gvs if "dense" in v.name]

        if loss_scale != 1:
            # When computing fp16 gradients, often intermediate tensor values are
            # so small, they underflow to 0. To avoid this, we multiply the loss by
            # loss_scale to make these tensor values loss_scale times bigger.
            scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

            if fine_tune:
                scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

            # Once the gradient computation is complete we can scale the gradients
            # back to the correct scale before passing them to the optimizer.
            unscaled_grad_vars = [
                (grad / loss_scale, var) for grad, var in scaled_grad_vars
            ]
            minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
        else:
            grad_vars = optimizer.compute_gradients(loss)
            if fine_tune:
                grad_vars = _dense_grad_filter(grad_vars)
            minimize_op = optimizer.apply_gradients(grad_vars, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(labels, predictions["classes"])
    accuracy_top_5 = tf.metrics.mean(
        tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name="top_5_op")
    )
    metrics = {"accuracy": accuracy, "accuracy_top_5": accuracy_top_5}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name="train_accuracy")
    tf.identity(accuracy_top_5[1], name="train_accuracy_top_5")
    tf.summary.scalar("train_accuracy", accuracy[1])
    tf.summary.scalar("train_accuracy_top_5", accuracy_top_5[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
    )


def cifar100_model_fn(features, labels, mode, params):
    """Model function for CIFAR-10."""
    # Learning rate schedule follows arXiv:1512.03385 for ResNet-56 and under.
    learning_rate_fn = learning_rate_with_decay(
        batch_size=params["batch_size"],
        batch_denom=128,
        num_images=_NUM_IMAGES["train"],
        boundary_epochs=[91, 136, 182],
        decay_rates=[1, 0.1, 0.01, 0.001],
    )

    # Weight decay of 2e-4 diverges from 1e-4 decay used in the ResNet paper
    # and seems more stable in testing. The difference was nominal for ResNet-56.
    weight_decay = 2e-4

    # Empirical testing showed that including batch_normalization variables
    # in the calculation of regularized loss helped validation accuracy
    # for the CIFAR-10 dataset, perhaps because the regularization prevents
    # overfitting on the small data set. We therefore include all vars when
    # regularizing and computing loss during training.
    def loss_filter_fn(_):
        return True

    return resnet_model_fn(
        features=features,
        labels=labels,
        mode=mode,
        model_class=ResNet18Cifar100,
        weight_decay=weight_decay,
        learning_rate_fn=learning_rate_fn,
        momentum=0.9,
        loss_scale=params["loss_scale"],
        loss_filter_fn=loss_filter_fn,
    )


def train(
    batch_size: int = 128,
    train_epochs: int = 182,
    epochs_between_evals: int = 10,
    multi_gpu: bool = False,
    label: str = None,
):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    tf.logging.set_verbosity(tf.logging.INFO)
    if label is None:
        model_dir = abspath("tf/resnet-18-cifar100/model_train/")
    else:
        model_dir = abspath(f"tf/resnet-18-cifar100/model_{label}/")
    # data_dir = abspath("/home/yxqiu/data/cifar100-raw")
    data_dir = abspath("/state/ssd0/yxqiu/data/cifar100-raw")

    model_function = cifar100_model_fn
    if multi_gpu:
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
        params={"batch_size": batch_size, "multi_gpu": multi_gpu, "loss_scale": 1},
    )

    for epoch in range(train_epochs // epochs_between_evals):
        # Train the model
        def train_input_fn():
            input = input_fn(
                is_training=True,
                data_dir=data_dir,
                batch_size=batch_size,
                num_epochs=epochs_between_evals,
            )
            return input

        # Set up training hook that logs the training accuracy every 100 steps.
        tensors_to_log = {"train_accuracy": "train_accuracy"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100
        )
        classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

        # Evaluate the model and print results
        def eval_input_fn():
            return input_fn(
                is_training=False,
                data_dir=data_dir,
                batch_size=batch_size,
                num_epochs=epochs_between_evals,
            )

        eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(label)
    print("Evaluation results:\n\t%s" % eval_results)
    print()


if __name__ == "__main__":
    fire.Fire(train)
