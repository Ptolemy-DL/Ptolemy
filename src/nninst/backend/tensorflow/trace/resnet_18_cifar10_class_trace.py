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

from nninst import mode
from nninst.backend.tensorflow.dataset.config import (
    CIFAR10_TRAIN,
    IMAGENET_RAW_TRAIN,
    IMAGENET_TRAIN,
)
from nninst.backend.tensorflow.model.config import RESNET_18_CIFAR10, RESNET_50
from nninst.backend.tensorflow.trace.common import (
    class_trace,
    class_trace_compact,
    class_trace_growth,
    full_trace,
    save_class_traces,
    save_class_traces_low_latency,
    save_full_trace_growth,
    self_similarity,
)
from nninst.utils.ray import ray_init

__all__ = ["resnet_18_cifar10_class_trace", "resnet_18_cifar10_self_similarity"]

name = "resnet_18_cifar10"

resnet_18_cifar10_class_trace = class_trace(
    name=name, model_config=RESNET_18_CIFAR10, data_config=CIFAR10_TRAIN
)

resnet_18_cifar10_class_trace_growth = class_trace_growth(
    name=name, model_config=RESNET_18_CIFAR10, data_config=CIFAR10_TRAIN
)

resnet_18_cifar10_class_trace_compact = class_trace_compact(
    resnet_18_cifar10_class_trace, name=name, model_config=RESNET_18_CIFAR10
)

save_resnet_18_cifar10_class_traces_low_latency = save_class_traces_low_latency(
    name=name, model_config=RESNET_18_CIFAR10, data_config=CIFAR10_TRAIN
)

resnet_18_cifar10_trace = full_trace(
    name=name, class_trace_fn=resnet_18_cifar10_class_trace
)

save_resnet_18_cifar10_trace_growth = save_full_trace_growth(
    name=name, class_trace_fn=resnet_18_cifar10_class_trace
)

resnet_18_cifar10_self_similarity = self_similarity(
    name=name, trace_fn=resnet_18_cifar10_class_trace, class_ids=range(0, 10)
)

if __name__ == "__main__":
    # mode.check(False)
    # mode.debug()
    # mode.local()
    mode.distributed()
    # ray_init("dell")
    # ray_init("gpu")
    ray_init()

    threshold = 0.5
    # threshold = 1
    # threshold = 0.8

    label = None
    # label = "train_50"
    # label = "train_start"
    # label = "train_start_more"

    # save_class_traces(resnet_18_cifar10_class_trace, range(0, 10), threshold=threshold, label=label,
    #                   example_num=5000, example_upperbound=5000,
    #                   )

    save_resnet_18_cifar10_class_traces_low_latency(
        range(0, 10), threshold=threshold, label=label, example_num=5000, batch_size=8
    )

    save_class_traces(
        resnet_18_cifar10_class_trace_compact,
        range(0, 10),
        threshold=threshold,
        label=label,
    )

    resnet_18_cifar10_self_similarity(threshold=threshold, label=label).save()
