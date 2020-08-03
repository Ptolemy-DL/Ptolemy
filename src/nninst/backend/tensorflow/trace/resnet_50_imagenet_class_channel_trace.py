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

from functools import partial

from nninst import mode
from nninst.backend.tensorflow.dataset.config import IMAGENET_RAW_TRAIN
from nninst.backend.tensorflow.model.config import RESNET_50
from nninst.backend.tensorflow.trace.common import (
    class_trace,
    class_trace_compact,
    full_trace,
    save_class_traces,
    save_full_trace_growth,
    self_similarity,
)
from nninst.channel_trace import get_channel_trace
from nninst.trace import merge_simple_trace
from nninst.utils.ray import ray_init

__all__ = [
    "resnet_50_imagenet_class_channel_trace",
    "resnet_50_imagenet_self_channel_similarity",
]

name = "resnet_50_imagenet"

resnet_50_imagenet_class_channel_trace = partial(
    class_trace(
        name=name, model_config=RESNET_50, data_config=IMAGENET_RAW_TRAIN, use_raw=True
    ),
    trace_fn=get_channel_trace,
    trace_type="class_channel_trace",
)

resnet_50_imagenet_class_channel_trace_compact = partial(
    class_trace_compact(
        resnet_50_imagenet_class_channel_trace, name=name, model_config=RESNET_50
    ),
    trace_type="class_channel_trace",
)

resnet_50_imagenet_channel_trace = full_trace(
    name=name, class_trace_fn=resnet_50_imagenet_class_channel_trace, per_channel=True
)

save_resnet_50_imagenet_channel_trace_growth = save_full_trace_growth(
    name=name, class_trace_fn=resnet_50_imagenet_class_channel_trace
)

resnet_50_imagenet_self_channel_similarity = self_similarity(
    name=name,
    trace_fn=resnet_50_imagenet_class_channel_trace,
    class_ids=range(1, 1001, 10),
    per_channel=True,
)

if __name__ == "__main__":
    # mode.check(False)
    mode.debug()
    # mode.local()
    # mode.distributed()
    # ray_init("dell")
    # ray_init("gpu")
    ray_init()

    # threshold = 0.5
    threshold = 1
    # threshold = 0.8

    label = None
    # label = "train_50"
    # label = "train_start"
    # label = "train_start_more"

    variant = None

    # print(f"generate class trace for label {label}")

    # save_class_traces(resnet_50_imagenet_class_trace, range(1, 1001), threshold=threshold, label=label)
    # save_resnet_50_imagenet_class_traces_low_latency(range(1, 2), threshold=threshold, label=label)

    # trace = resnet_50_imagenet_class_trace(11, threshold=threshold, label="train_40036").load()

    steps = [500450, 200180, 80072, 60054, 40036, 20018, 1000901, 2362125]

    save_class_traces(
        resnet_50_imagenet_class_channel_trace,
        range(1, 1001),
        threshold=threshold,
        label=label,
        example_num=100,
        example_upperbound=1000,
        parallel=1,
        variant=variant,
        # merge_fn=merge_simple_trace_intersect,
        merge_fn=merge_simple_trace,
    )

    save_class_traces(
        resnet_50_imagenet_class_channel_trace_compact,
        range(1, 1001),
        threshold=threshold,
        label=label,
        variant=variant,
    )

    resnet_50_imagenet_channel_trace(
        threshold=threshold, label=label, class_ids=range(1, 1001), variant=variant
    ).save()

    resnet_50_imagenet_self_channel_similarity(threshold, label, variant=variant).save()

    # for step in steps:
    #     label = f"train_{step}"
    #     print(f"generate class trace for label {label}")
    #     save_class_traces(resnet_50_imagenet_class_trace, range(1, 1001, 10), threshold=threshold, label=label,
    #                       example_num=100, example_upperbound=1000)
    #
    # for step in steps:
    #     label = f"train_{step}"
    #     print(f"generate compact class trace for label {label}")
    #     save_class_traces(resnet_50_imagenet_class_trace_compact, range(1, 1001, 10), threshold=threshold, label=label)
    #
    # for step in steps:
    #     label = f"train_{step}"
    #     print(f"generate full trace for label {label}")
    #     resnet_50_imagenet_trace(threshold=threshold, label=label, class_ids=range(1, 1001, 10)).save()
    #
    # for step in steps:
    #     label = f"train_{step}"
    #     print(f"generate self-similarity matrix for label {label}")
    #     # resnet_50_imagenet_self_similarity(threshold, label).save()
    #     resnet_50_imagenet_self_channel_similarity(threshold, label, key=TraceKey.WEIGHT).save()

    # for step in steps:
    #     label = f"train_{step}"
    #     trace = resnet_50_imagenet_trace(threshold=threshold, label=label).load()
    #     for key in [TraceKey.EDGE, TraceKey.WEIGHT]:
    #         print(f"{label} {key}: {calc_density_compact(trace, key)}")

    # resnet_50_imagenet_trace(threshold=threshold, label=label, class_ids=range(1, 1001)).save()
    # resnet_50_imagenet_trace(threshold=threshold, label=label, class_ids=range(1, 1001, 10)).save()

    # check_class_traces(resnet_50_imagenet_class_trace, range(1, 1001),
    #                    threshold=threshold, label="compact", compress=True)

    # trace = resnet_50_imagenet_class_trace(1, threshold=threshold, label=label).load()
    # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    #     print(f"{key}: {calc_density(trace, key)}")
    # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    #     print(f"{key}: {calc_space(trace, key)}")

    # trace = resnet_50_imagenet_trace(threshold=threshold, label=label).load()
    # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    #     print(f"{key}: {calc_density_compact(trace, key)}")

    # resnet_50_imagenet_self_similarity(threshold, label).save()
    # similarity_matrix = resnet_50_imagenet_self_similarity(threshold, label).load()

    # save_resnet_50_imagenet_trace_growth(threshold=threshold, label=label, class_ids=range(506, 1001),
    #                                      start_from=range(1, 506))

    # trace = resnet_50_imagenet_trace(threshold=threshold, label=label).load()
    # layers = ResNet50.graph().load().layers()
    # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    #     density_per_layer = calc_density_compact_per_layer(trace, layers, key)
    #     density_per_layer.to_csv(abspath(f"resnet_50_imagenet_trace_per_layer.{key}.csv"))

    # trace = resnet_50_imagenet_trace(threshold=threshold, label=label).load()
    # graph = ResNet50.graph().load()
    # graph.load_attrs(trace)
    # layers = ResNet50.graph().load().layers()
    # skip_ratio = calc_skip_ratio(graph, layers)
    # skip_ratio.to_csv(abspath(f"resnet_50_imagenet_skip_ratio.csv"))
