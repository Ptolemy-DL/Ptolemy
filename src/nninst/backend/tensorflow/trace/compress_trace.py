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

import numpy as np
import ray

from nninst import AttrMap, mode
from nninst.backend.tensorflow.model import ResNet50
from nninst.backend.tensorflow.trace.resnet_50_imagenet_class_trace_v3 import (
    resnet_50_imagenet_class_trace,
)
from nninst.trace import compact_trace
from nninst.utils.fs import IOAction
from nninst.utils.ray import ray_init, ray_map

__all__ = ["resnet_50_imagenet_class_trace"]


@ray.remote
class ClassTrace:
    def get_trace(
        self, class_id: int, threshold: float, label: str = None, compress: bool = False
    ) -> AttrMap:
        return resnet_50_imagenet_class_trace(
            class_id, threshold, label, compress=compress
        ).load()


def resnet_50_imagenet_class_trace_compress(
    class_id: int, threshold: float, label: str, *trace_actors: ClassTrace
) -> IOAction[AttrMap]:
    def compress():
        # return resnet_50_imagenet_class_trace(class_id, threshold, label).load()
        return ray.get(
            trace_actors[class_id % len(trace_actors)].get_trace.remote(
                class_id, threshold, label
            )
        )

    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "resnet_50_imagenet_compress"
    else:
        name = f"resnet_50_imagenet_{label}_compress"
    path = f"store/analysis/class_trace/{name}/approx_{threshold_name}/{class_id}.pkl"
    return IOAction(path, init_fn=compress, cache=True, compress=True)


def resnet_50_imagenet_class_trace_compact(
    class_id: int, threshold: float, label: str, *trace_actors: ClassTrace
) -> IOAction[AttrMap]:
    def compact():
        return compact_trace(
            ray.get(
                trace_actors[class_id % len(trace_actors)].get_trace.remote(
                    class_id,
                    threshold,
                    "compress" if label is None else label + "_compress",
                    compress=True,
                )
            ),
            ResNet50.graph().load(),
        )

    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "resnet_50_imagenet_compact"
    else:
        name = f"resnet_50_imagenet_{label}_compact"
    path = f"store/analysis/class_trace/{name}/approx_{threshold_name}/{class_id}.pkl"
    return IOAction(path, init_fn=compact, cache=True, compress=True)


if __name__ == "__main__":
    # mode.check(False)
    # mode.debug()
    mode.local()
    # mode.distributed()
    ray_init("gpu", num_cpus=7)
    threshold = 0.5
    # threshold = 1
    # threshold = 0.8

    # label = None
    label = "train_50"

    print(f"generate class trace for label {label}")

    trace_actors = [ClassTrace.remote() for _ in range(2)]

    should_stop = False

    while not should_stop:

        def compress(class_id: int, *trace_actors):
            original_trace = resnet_50_imagenet_class_trace(class_id, threshold, label)
            if original_trace.is_saved():
                try:
                    resnet_50_imagenet_class_trace_compress(
                        class_id, threshold, label, *trace_actors
                    ).save()
                    return True
                except Exception as cause:
                    return False
                    # raise RuntimeError(f"raise from compress, class {class_id}") from cause
            else:
                return False

        result = ray_map(
            compress,
            [[class_id] + trace_actors for class_id in range(1, 1001)],
            chunksize=1,
            out_of_order=True,
            num_gpus=0,
            huge_task=True,
            pass_actor=True,
        )
        should_stop = np.all(np.array(result))

        def compact(class_id: int, *trace_actors):
            compressed_trace = resnet_50_imagenet_class_trace(
                class_id,
                threshold,
                "compress" if label is None else label + "_compress",
                compress=True,
            )
            if compressed_trace.is_saved():
                try:
                    resnet_50_imagenet_class_trace_compact(
                        class_id, threshold, label, *trace_actors
                    ).save()
                    return True
                # except EOFError:
                except Exception:
                    os.remove(compressed_trace.path)
                    return False
                # except Exception as cause:
                #     raise RuntimeError(f"raise from compact, class {class_id}") from cause
            else:
                return False

        result = ray_map(
            compact,
            [[class_id] + trace_actors for class_id in range(1, 1001)],
            chunksize=1,
            out_of_order=True,
            num_gpus=0,
            huge_task=True,
            pass_actor=True,
        )

        should_stop = should_stop and np.all(np.array(result))
