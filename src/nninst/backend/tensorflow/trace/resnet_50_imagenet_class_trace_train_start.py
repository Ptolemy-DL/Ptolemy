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
from functools import partial

import numpy as np
import tensorflow as tf

from nninst import AttrMap, mode
from nninst.backend.tensorflow.dataset import imagenet
from nninst.backend.tensorflow.graph import model_fn_with_fetch_hook
from nninst.backend.tensorflow.model import ResNet50
from nninst.backend.tensorflow.trace.common import reconstruct_class_trace_from_tf
from nninst.statistics import calc_iou, self_similarity_matrix
from nninst.trace import merge_simple_trace
from nninst.utils.fs import IOAction, abspath
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init, ray_map_reduce

__all__ = ["resnet_50_imagenet_class_trace", "resnet_50_imagenet_self_similarity"]


def resnet_50_imagenet_class_trace(
    class_id: int,
    threshold: float,
    batch_size: int = 256,
    label: str = None,
    cache: bool = False,
) -> IOAction[AttrMap]:
    def get_class_trace() -> AttrMap:
        try:
            mode.check(False)
            data_dir = abspath("/home/yxqiu/data/imagenet")
            # data_dir = abspath("/state/ssd0/yxqiu/data/imagenet")
            class_trace = reconstruct_class_trace_from_tf(
                class_id,
                model_fn=partial(
                    model_fn_with_fetch_hook,
                    create_model=lambda: ResNet50(),
                    graph=ResNet50.graph().load(),
                ),
                input_fn=lambda: imagenet.train(
                    data_dir,
                    batch_size,
                    transform_fn=lambda dataset: dataset.filter(
                        lambda image, label: tf.equal(
                            tf.convert_to_tensor(class_id, dtype=tf.int32), label
                        )
                    ).take(140),
                ),
                # ).skip(3).take(1)),
                model_dir=abspath("tf/resnet-50-v2/model/"),
                # model_dir=abspath("/state/ssd0/yxqiu/workspace/nninst/tf/resnet-50-v2/model/"),
                select_fn=lambda input: arg_approx(input, threshold),
                parallel=4,
            )
            return class_trace
        except Exception as cause:
            raise RuntimeError(f"error when handling class {class_id}") from cause

    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "resnet_50_imagenet"
    else:
        name = f"resnet_50_imagenet_{label}"
    path = f"store/analysis/class_trace/{name}/approx_{threshold_name}/{class_id}.pkl"
    return IOAction(path, init_fn=get_class_trace, cache=cache)


def save_all_resnet_50_imagenet_class_traces(threshold: float, label: str = None):
    def get_trace(class_id: int, batch_id: int, batch_size: int = 1) -> AttrMap:
        try:
            mode.check(False)
            data_dir = abspath("/home/yxqiu/data/imagenet")
            trace = reconstruct_class_trace_from_tf(
                model_fn=partial(
                    model_fn_with_fetch_hook,
                    create_model=lambda: ResNet50(),
                    graph=ResNet50.graph().load(),
                ),
                input_fn=lambda: imagenet.train(
                    data_dir,
                    batch_size,
                    transform_fn=lambda dataset: dataset.filter(
                        lambda image, label: tf.equal(
                            tf.convert_to_tensor(class_id, dtype=tf.int32), label
                        )
                    )
                    .skip(batch_id * batch_size)
                    .take(batch_size),
                ),
                model_dir=abspath("tf/resnet-50-v2/model_train_start"),
                select_fn=lambda input: arg_approx(input, threshold),
                class_id=class_id,
                parallel=4,
            )
            # trace = AttrMap()
            return trace
        except Exception as cause:
            raise RuntimeError(
                f"error when handling class {class_id} batch {batch_id}"
            ) from cause

    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "resnet_50_imagenet"
    else:
        name = f"resnet_50_imagenet_{label}"
    prefix = f"store/analysis/class_trace/{name}/approx_{threshold_name}"
    class_ids = [
        class_id
        for class_id in range(1, 2)
        if not os.path.exists(f"{prefix}/{class_id}.pkl")
    ]
    # image_num = 140
    image_num = 530
    batch_size = 1
    batch_num = image_num // batch_size
    traces = ray_map_reduce(
        get_trace,
        merge_simple_trace,
        [
            (
                class_id,
                [(class_id, batch_id, batch_size) for batch_id in range(0, batch_num)],
            )
            for class_id in class_ids
        ],
        num_gpus=0,
    )
    for class_id, trace in traces:
        IOAction(
            f"{prefix}/{class_id}.pkl", init_fn=lambda: trace, compress=True
        ).save()
        print(f"finish class {class_id}")
    # merged_traces = defaultdict(lambda: None)
    # trace_count = defaultdict(int)
    # for next_trace in traces:
    #     class_id, batch_id, trace = next_trace
    #     print(f"begin class {class_id} batch {batch_id}")
    #     if trace is not None:
    #         merged_traces[class_id] = merge_trace(trace, merged_traces[class_id])
    #     # print(f"finish class {class_id} image {batch_id}")
    #     trace_count[class_id] = trace_count[class_id] + 1
    #     if trace_count[class_id] == batch_num:
    #         IOAction(f"{prefix}/{class_id}.pkl", init_fn=lambda: merged_traces[class_id]).save()
    #         del trace_count[class_id]
    #         print(f"============== finish class {class_id}")


def resnet_50_imagenet_self_similarity(
    threshold: float, label: str = None
) -> IOAction[np.ndarray]:
    def get_self_similarity() -> np.ndarray:
        return self_similarity_matrix(
            range(1000),
            trace_fn=lambda class_id: resnet_50_imagenet_class_trace(
                class_id, threshold, label
            ).load(),
            similarity_fn=calc_iou,
        )

    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "resnet_50_imagenet"
    else:
        name = f"resnet_50_imagenet_{label}"
    path = f"store/analysis/self_similarity/{name}/approx_{threshold_name}/self_similarity.pkl"
    return IOAction(path, init_fn=get_self_similarity)


if __name__ == "__main__":
    # mode.check(False)
    # mode.debug()
    mode.local()
    # mode.distributed()
    ray_init("dell")
    threshold = 0.5
    # threshold = 1
    # threshold = 0.8

    # label = "train_start"
    label = "train_start_more"

    print(f"generate class trace for label {label}")

    save_all_resnet_50_imagenet_class_traces(threshold, label)
