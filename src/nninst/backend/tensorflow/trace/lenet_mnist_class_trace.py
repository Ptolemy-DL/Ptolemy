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

from functools import partial, reduce
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from nninst import AttrMap, GraphAttrKey, mode
from nninst.backend.tensorflow.dataset import mnist
from nninst.backend.tensorflow.graph import model_fn_with_fetch_hook
from nninst.backend.tensorflow.model import LeNet
from nninst.backend.tensorflow.trace.common import (
    reconstruct_static_trace_from_tf,
    reconstruct_trace_from_tf,
)
from nninst.dataset import mnist_info
from nninst.statistics import calc_density_compact, calc_iou, self_similarity_matrix
from nninst.trace import TraceKey, compact_trace, merge_trace
from nninst.utils import filter_not_null, grouper
from nninst.utils.fs import IOAction, abspath
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_iter

__all__ = [
    "lenet_mnist_class_trace",
    "lenet_mnist_trace",
    "lenet_mnist_self_similarity",
]


def lenet_mnist_class_trace(
    class_id: int, threshold: float, batch_size: int = 256, label: str = None
) -> IOAction[AttrMap]:
    def get_class_trace() -> AttrMap:
        def get_trace(image_ids: Tuple[int, ...]) -> Optional[AttrMap]:
            try:
                mode.check(False)
                data_dir = abspath("/home/yxqiu/data/mnist/raw")
                batched_traces = reconstruct_trace_from_tf(
                    model_fn=partial(
                        model_fn_with_fetch_hook,
                        create_model=lambda: LeNet(),
                        graph=LeNet.graph().load(),
                    ),
                    input_fn=lambda: tf.data.Dataset.zip(
                        (tf.data.Dataset.range(60000), mnist.train(data_dir))
                    )
                    .filter(
                        lambda image_id, data: tf.reduce_any(
                            tf.equal(
                                tf.convert_to_tensor(image_ids, dtype=tf.int64),
                                image_id,
                            )
                        )
                    )
                    .map(lambda image_id, data: data)
                    .batch(batch_size),
                    select_fn=lambda input: arg_approx(input, threshold),
                    model_dir=abspath("tf/lenet/model/"),
                )
                batched_traces = [
                    trace
                    for trace in batched_traces
                    if trace.attrs[GraphAttrKey.PREDICT] == class_id
                ]
                return merge_trace(*batched_traces)
            except Exception as cause:
                raise RuntimeError(
                    f"error when handling class {class_id} image {image_ids}"
                ) from cause

        traces = ray_iter(
            get_trace,
            grouper(
                batch_size,
                (
                    image_id
                    for image_id in range(mnist_info.train().size)
                    if mnist_info.train().label(image_id) == class_id
                ),
            ),
            chunksize=1,
            out_of_order=True,
            num_gpus=0,
        )
        class_trace = reduce(merge_trace, filter_not_null(traces))
        return class_trace

    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "lenet_mnist"
    else:
        name = f"lenet_mnist_{label}"
    path = f"store/analysis/class_trace/{name}/approx_{threshold_name}/{class_id}.pkl"
    return IOAction(path, init_fn=get_class_trace)


def lenet_mnist_trace(threshold: float, label: str = None) -> IOAction[AttrMap]:
    def get_trace() -> AttrMap:
        return merge_trace(
            *[
                lenet_mnist_class_trace(
                    class_id, threshold=threshold, label=label
                ).load()
                for class_id in range(0, 10)
            ]
        )

    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "lenet_mnist"
    else:
        name = f"lenet_mnist_{label}"
    path = f"store/analysis/trace/{name}/approx_{threshold_name}/trace.pkl"
    return IOAction(path, init_fn=get_trace)


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


def lenet_mnist_self_similarity(
    threshold: float, label: str = None
) -> IOAction[np.ndarray]:
    def get_self_similarity() -> np.ndarray:
        return self_similarity_matrix(
            range(10),
            trace_fn=lambda class_id: lenet_mnist_class_trace(
                class_id, threshold, label
            ).load(),
            similarity_fn=calc_iou,
        )

    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "lenet_mnist"
    else:
        name = f"lenet_mnist_{label}"
    path = f"store/analysis/self_similarity/{name}/approx_{threshold_name}/self_similarity.pkl"
    return IOAction(path, init_fn=get_self_similarity)


if __name__ == "__main__":
    # mode.check(False)
    # mode.debug()
    # mode.local()
    mode.distributed()
    # ray_init()
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
    # for class_id in range(10):
    #     lenet_mnist_class_trace(class_id, threshold=threshold, label=label).save()

    # lenet_mnist_class_trace(class_id, threshold=threshold).save()
    # trace = class_trace(class_id, threshold=threshold).load()
    # print(trace)

    # lenet_mnist_self_similarity(threshold, label="early").save()
    # similarity = lenet_mnist_self_similarity(threshold).load()
    # print(similarity)

    # lenet_mnist_trace(threshold, label).save()

    # trace = lenet_mnist_trace(threshold, label).load()
    # trace = lenet_mnist_class_trace(0, threshold, label=label).load()
    # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    #     print(f"{key}: {calc_density(trace, key)}")
    # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    #     print(f"{key}: {calc_space(trace, key)}")

    trace = compact_trace(
        lenet_mnist_trace(threshold, label).load(), LeNet.graph().load()
    )
    for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
        print(f"{key}: {calc_density_compact(trace, key)}")

    # dynamic_trace = lenet_mnist_trace(threshold, label).load()

    # lenet_mnist_static_trace(threshold, label).save()

    # static_trace = lenet_mnist_static_trace(threshold, label).load()
    # key = TraceKey.WEIGHT
    # print(f"{key}: {calc_density(static_trace, key)}")

    # print(f"iou(dynamic, static): {calc_iou(dynamic_trace, static_trace, key=TraceKey.WEIGHT)}")
