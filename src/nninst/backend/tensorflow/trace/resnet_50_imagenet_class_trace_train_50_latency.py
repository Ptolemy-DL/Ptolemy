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

import traceback
from functools import partial
from typing import Tuple, Union

import numpy as np
import tensorflow as tf

from nninst import AttrMap, mode
from nninst.backend.tensorflow.dataset import imagenet
from nninst.backend.tensorflow.graph import model_fn_with_fetch_hook
from nninst.backend.tensorflow.model import ResNet50
from nninst.backend.tensorflow.trace.common import reconstruct_class_trace_from_tf
from nninst.utils.fs import IOAction, abspath
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init, ray_iter

__all__ = ["resnet_50_imagenet_class_trace"]


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
                model_dir=abspath("tf/resnet-50-v2/model_train_50/"),
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


if __name__ == "__main__":
    # mode.check(False)
    # mode.debug()
    # mode.local()
    mode.distributed()
    ray_init("dell")
    threshold = 0.5
    # threshold = 1
    # threshold = 0.8

    label = "train_50"

    print(f"generate class trace for label {label}")

    def class_trace(class_id: int) -> Union[int, Tuple[int, str]]:
        try:
            resnet_50_imagenet_class_trace(
                class_id, threshold=threshold, batch_size=16, label=label, cache=True
            ).save()
            # resnet_50_imagenet_class_trace(class_id, threshold=threshold, batch_size=1, label=label).save()
            return class_id
        except Exception:
            return class_id, traceback.format_exc()

    results = ray_iter(
        class_trace,
        np.arange(1, 1001),
        chunksize=1,
        out_of_order=True,
        num_gpus=0,
        huge_task=True,
    )
    # ray_map(class_trace, [454], chunksize=1, out_of_order=True, num_gpus=1)
    for result in results:
        if not isinstance(result, int):
            class_id, tb = result
            print(f"## raise exception from class {class_id}:")
            print(tb)
        else:
            print(f"finish class {result}")
    print("finish")
    # ray_map(class_trace, range(1, 1001), chunksize=1, out_of_order=True, num_gpus=1)
    # for class_id in range(10):
    #     resnet_50_imagenet_class_trace(class_id, threshold=threshold, label=label).save()
    # lenet_mnist_class_trace(class_id, threshold=threshold).save()
    # trace = class_trace(class_id, threshold=threshold).load()
    # print(trace)

    # lenet_mnist_self_similarity(threshold, label="early").save()
    # similarity = lenet_mnist_self_similarity(threshold).load()
    # print(similarity)
