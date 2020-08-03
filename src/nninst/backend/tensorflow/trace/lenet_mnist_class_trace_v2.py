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

import tensorflow as tf

from nninst import AttrMap, mode
from nninst.backend.tensorflow.dataset import mnist
from nninst.backend.tensorflow.dataset.config import MNIST_TRAIN
from nninst.backend.tensorflow.model import LeNet
from nninst.backend.tensorflow.model.config import LENET
from nninst.backend.tensorflow.trace.common import (
    class_trace,
    class_trace_compact,
    full_trace,
    reconstruct_static_trace_from_tf,
    save_class_traces,
    self_similarity,
)
from nninst.statistics import calc_density_compact
from nninst.trace import TraceKey
from nninst.utils.fs import IOAction, abspath
from nninst.utils.ray import ray_init

__all__ = [
    "lenet_mnist_class_trace",
    "lenet_mnist_trace",
    "lenet_mnist_self_similarity",
]

name = "lenet_mnist"

lenet_mnist_class_trace = class_trace(name, model_config=LENET, data_config=MNIST_TRAIN)

lenet_mnist_class_trace_compact = class_trace_compact(
    lenet_mnist_class_trace, name=name, model_config=LENET
)

lenet_mnist_trace = full_trace(name, lenet_mnist_class_trace)

threshold_to_density = {0.5: 0.1379723919438787}


def lenet_mnist_static_trace(threshold: float, label: str = None) -> IOAction[AttrMap]:
    def get_trace() -> AttrMap:
        return reconstruct_static_trace_from_tf(
            model_fn=lambda: LeNet(),
            input_fn=lambda: tf.placeholder(tf.float32, shape=(1, 1, 28, 28)),
            model_dir=tf.train.latest_checkpoint(abspath("tf/lenet/model/")),
            density=threshold_to_density[threshold],
        )

    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "lenet_mnist"
    else:
        name = f"lenet_mnist_{label}"
    path = f"store/analysis/static_trace/{name}/approx_{threshold_name}/trace.pkl"
    return IOAction(path, init_fn=get_trace)


lenet_mnist_self_similarity = self_similarity(
    name, lenet_mnist_class_trace, range(0, 10)
)

if __name__ == "__main__":
    # mode.check(False)
    # mode.debug()
    # mode.local()
    mode.distributed()
    # ray_init("gpu")
    ray_init()
    threshold = 0.5
    # threshold = 1
    # threshold = 0.8

    label = "early"
    # label = "best_in_10"
    # label = "worst_in_10"
    # label = "import"
    # label = "norm"
    # label = "test"

    print(f"generate class trace for label {label}")

    for threshold in [
        # 1.0,
        # 0.9,
        # 0.7,
        0.5,
        # 0.3,
        # 0.1,
    ]:
        save_class_traces(
            lenet_mnist_class_trace,
            range(10),
            threshold=threshold,
            label=label,
            batch_size=256,
            example_num=6000,
            example_upperbound=6000,
        )
        # save_class_traces(lenet_mnist_class_trace, range(0, 1), threshold=threshold, label=label,
        #                   example_num=100, example_upperbound=1000)

        save_class_traces(
            lenet_mnist_class_trace_compact, range(10), threshold=threshold, label=label
        )

    # lenet_mnist_self_similarity(threshold=threshold, label=label, compress=False).save()

    lenet_mnist_trace(class_ids=range(0, 10), threshold=threshold, label=label).save()

    trace = lenet_mnist_trace(threshold, label).load()
    for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
        print(f"{key}: {calc_density_compact(trace, key)}")
    # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    #     print(f"{key}: {calc_space(trace, key)}")

    # trace = compact_trace(lenet_mnist_trace(threshold, label).load(), LeNet.graph().load())
    # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    #     print(f"{key}: {calc_density_compact(trace, key)}")

    # dynamic_trace = lenet_mnist_trace(threshold, label).load()

    # lenet_mnist_static_trace(threshold, label).save()

    # static_trace = lenet_mnist_static_trace(threshold, label).load()
    # key = TraceKey.WEIGHT
    # print(f"{key}: {calc_density(static_trace, key)}")

    # print(f"iou(dynamic, static): {calc_iou(dynamic_trace, static_trace, key=TraceKey.WEIGHT)}")

    # for class_id in range(10):
    #     trace = lenet_mnist_class_trace(class_id, threshold, label, compress=False).load()
    #     threshold_name = "{0:.3f}".format(threshold)
    #     trace_name = f"{name}_{label}_compress"
    #     IOAction(f"store/analysis/class_trace/{trace_name}/approx_{threshold_name}/{class_id}.pkl",
    #              init_fn=lambda: trace, compress=True).save()

    # trace = lenet_mnist_trace(threshold=threshold, label=label).load()
    # layers = LeNet.graph().load().layers()
    # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    #     density_per_layer = calc_density_compact_per_layer(trace, layers, key)
    #     density_per_layer.to_csv(abspath(f"lenet_mnist_trace_per_layer.{key}.csv"))
