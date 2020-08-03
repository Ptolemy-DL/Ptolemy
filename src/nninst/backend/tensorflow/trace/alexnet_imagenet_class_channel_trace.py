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
from nninst.backend.tensorflow.dataset import imagenet_raw
from nninst.backend.tensorflow.dataset.config import IMAGENET_RAW_TRAIN
from nninst.backend.tensorflow.dataset.imagenet_preprocessing import (
    alexnet_preprocess_image,
)
from nninst.backend.tensorflow.model.config import ALEXNET
from nninst.backend.tensorflow.trace.common import (
    class_trace,
    class_trace_compact,
    full_intersect_trace,
    full_trace,
    save_class_traces,
    self_similarity,
)
from nninst.channel_trace import get_channel_trace
from nninst.statistics import calc_trace_size
from nninst.trace import (
    merge_compact_trace_xor,
    merge_simple_trace,
    merge_simple_trace_xor,
)
from nninst.utils.ray import ray_init

__all__ = ["alexnet_imagenet_class_channel_trace", "alexnet_imagenet_self_similarity"]


def dataset_fn(*args, **kwargs):
    return imagenet_raw.train(
        *args, **kwargs, class_from_zero=True, preprocessing_fn=alexnet_preprocess_image
    )


data_config = IMAGENET_RAW_TRAIN.copy(dataset_fn=dataset_fn)

name = "alexnet_imagenet"

alexnet_imagenet_class_channel_trace = partial(
    class_trace(name=name, model_config=ALEXNET, data_config=data_config, use_raw=True),
    trace_fn=get_channel_trace,
    trace_type="class_channel_trace",
)

alexnet_imagenet_class_channel_trace_compact = partial(
    class_trace_compact(
        alexnet_imagenet_class_channel_trace, name=name, model_config=ALEXNET
    ),
    trace_type="class_channel_trace",
)

alexnet_imagenet_channel_trace = full_trace(
    name=name, class_trace_fn=alexnet_imagenet_class_channel_trace, per_channel=True
)

alexnet_imagenet_channel_intersect_trace = full_intersect_trace(
    name=name, class_trace_fn=alexnet_imagenet_class_channel_trace, per_channel=True
)

alexnet_imagenet_self_similarity = self_similarity(
    name=name,
    trace_fn=alexnet_imagenet_class_channel_trace,
    class_ids=range(0, 1000, 10),
    per_channel=True,
)

if __name__ == "__main__":
    # mode.check(False)
    # mode.debug()
    # mode.local()
    mode.distributed()
    # ray_init("dell")
    ray_init()
    # ray_init("r730")
    # threshold = 0.5
    # threshold = 0.3
    # threshold = 0.1
    threshold = 1
    # threshold = 0.8

    label = "import"
    # label = "import_old"

    # variant = "intersect"
    # variant = "regen"
    variant = None

    print(f"generate class trace for label {label}")

    for threshold in [
        1,
        # 0.5,
        # 0.3,
        # 0.9,
    ]:
        save_class_traces(
            partial(
                alexnet_imagenet_class_channel_trace,
                threshold=threshold,
                label=label,
                example_num=100,
                example_upperbound=1000,
                parallel=4,
                variant=variant,
                # merge_fn=merge_simple_trace_intersect,
                merge_fn=merge_simple_trace,
            ),
            range(0, 1000, 10),
        )

        save_class_traces(
            partial(
                alexnet_imagenet_class_channel_trace_compact,
                threshold=threshold,
                label=label,
                variant=variant,
            ),
            range(0, 1000, 10),
        )

        # alexnet_imagenet_channel_trace(
        #     threshold=threshold, label=label, class_ids=range(0, 1000), variant=variant).save()

        # check_class_traces(alexnet_imagenet_class_trace, range(0, 1000), threshold, label, compress=True)

        alexnet_imagenet_self_similarity(threshold, label, variant=variant).save()

    # trace = alexnet_imagenet_channel_trace(threshold, label).load()
    # trace = alexnet_imagenet_class_channel_trace_compact(0, threshold, label, variant=variant).load()
    # for key in [TraceKey.POINT,
    #             TraceKey.EDGE,
    #             # TraceKey.WEIGHT,
    #             ]:
    #     print(f"{key}: {calc_density_compact(trace, key)}")
    # for key in [
    #     TraceKey.POINT,
    #     TraceKey.EDGE,
    #     # TraceKey.WEIGHT,
    # ]:
    #     print(f"{key}: {calc_space(trace, key)}")

    # for class_id in range(0, 1000, 10):
    #     trace = alexnet_imagenet_class_channel_trace_compact(class_id, threshold, label).load()
    #     regen_trace = alexnet_imagenet_class_channel_trace_compact(class_id, threshold, label, variant="regen").load()
    #     xor_trace = merge_compact_trace_xor(trace, regen_trace)
    #     xor_size = calc_trace_size(xor_trace, compact=True)
    #     print(xor_size)

    # for class_id in range(0, 1000, 10):
    #     trace = alexnet_imagenet_class_channel_trace(class_id, threshold, label).load()
    #     regen_trace = alexnet_imagenet_class_channel_trace(class_id, threshold, label, variant="regen").load()
    #     xor_trace = merge_simple_trace_xor(trace, regen_trace)
    #     xor_size = calc_trace_size(xor_trace)
    #     print(xor_size)
