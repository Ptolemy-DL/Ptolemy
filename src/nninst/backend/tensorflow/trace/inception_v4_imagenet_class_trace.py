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
from nninst.backend.tensorflow.dataset import imagenet_raw
from nninst.backend.tensorflow.dataset.config import IMAGENET_RAW_TRAIN
from nninst.backend.tensorflow.dataset.imagenet import inception_preprocess_image
from nninst.backend.tensorflow.model.config import INCEPTION_V4
from nninst.backend.tensorflow.trace.common import (
    class_trace,
    class_trace_compact,
    full_trace,
    save_class_traces_low_latency,
    self_similarity,
)
from nninst.statistics import calc_density_compact
from nninst.trace import TraceKey
from nninst.utils.ray import ray_init

__all__ = ["inception_v4_imagenet_class_trace", "inception_v4_imagenet_self_similarity"]


def dataset_fn(*args, **kwargs):
    return imagenet_raw.train(
        *args, **kwargs, image_size=299, preprocessing_fn=inception_preprocess_image
    )


data_config = IMAGENET_RAW_TRAIN.copy(dataset_fn=dataset_fn)

name = "inception_v4_imagenet"

inception_v4_imagenet_class_trace = class_trace(
    name=name, model_config=INCEPTION_V4, data_config=data_config, use_raw=True
)

inception_v4_imagenet_class_trace_compact = class_trace_compact(
    inception_v4_imagenet_class_trace, name=name, model_config=INCEPTION_V4
)

save_inception_v4_imagenet_class_traces_low_latency = save_class_traces_low_latency(
    name=name, model_config=INCEPTION_V4, data_config=data_config, use_raw=True
)

inception_v4_imagenet_trace = full_trace(
    name=name, class_trace_fn=inception_v4_imagenet_class_trace
)

inception_v4_imagenet_self_similarity = self_similarity(
    name=name, trace_fn=inception_v4_imagenet_class_trace, class_ids=range(1, 1001, 10)
)

if __name__ == "__main__":
    # mode.check(False)
    # mode.debug()
    # mode.local()
    mode.distributed()
    # ray_init("r730")
    # ray_init("dell")
    ray_init()
    threshold = 0.5
    # threshold = 1
    # threshold = 0.8

    label = None

    print(f"generate class trace for label {label}")

    # trace = inception_v4_imagenet_class_trace(103, threshold=threshold, label=label).load()

    # save_class_traces(inception_v4_imagenet_class_trace, range(1, 1001), threshold=threshold, label=label,
    #                   example_num=100, example_upperbound=1000)

    # save_class_traces(
    #     inception_v4_imagenet_class_trace,
    #     np.transpose(np.reshape(np.array(range(1, 1001)), (100, 10))).flatten()[500:],
    #     # [603],
    #     threshold=threshold,
    #     label=label,
    #     example_num=100,
    #     example_upperbound=1000,
    #     parallel=4,
    # )

    # save_class_traces(inception_v4_imagenet_class_trace_compact, range(1, 1001), threshold=threshold, label=label)

    # inception_v4_imagenet_trace(threshold=threshold, label=label, class_ids=range(1, 1001)).save()

    # inception_v4_imagenet_self_similarity(threshold, label).save()

    # trace = inception_v4_imagenet_class_trace(1, threshold, label, 1, 1000).init_fn()
    # compact_trace(trace, InceptionV4.graph().load())
    # print()

    # check_class_traces(
    #     inception_v4_imagenet_class_trace,
    #     range(1, 1001),
    #     threshold=threshold,
    #     label=label,
    #     compress=True,
    # )

    trace = inception_v4_imagenet_trace(threshold, label).load()
    for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
        print(f"{key}: {calc_density_compact(trace, key)}")
