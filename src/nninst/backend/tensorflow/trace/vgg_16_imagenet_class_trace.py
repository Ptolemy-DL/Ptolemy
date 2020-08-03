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
from nninst.backend.tensorflow.model.config import VGG_16
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

__all__ = ["vgg_16_imagenet_class_trace", "vgg_16_imagenet_self_similarity"]


def dataset_fn(*args, **kwargs):
    return imagenet_raw.train(*args, **kwargs, class_from_zero=True)


data_config = IMAGENET_RAW_TRAIN.copy(dataset_fn=dataset_fn)

name = "vgg_16_imagenet"

vgg_16_imagenet_class_trace = class_trace(
    name=name, model_config=VGG_16, data_config=data_config, use_raw=True
)

vgg_16_imagenet_class_trace_compact = class_trace_compact(
    vgg_16_imagenet_class_trace, name=name, model_config=VGG_16
)

save_vgg_16_imagenet_class_traces_low_latency = save_class_traces_low_latency(
    name=name, model_config=VGG_16, data_config=data_config, use_raw=True
)

vgg_16_imagenet_trace = full_trace(
    name=name, class_trace_fn=vgg_16_imagenet_class_trace
)

vgg_16_imagenet_self_similarity = self_similarity(
    name=name, trace_fn=vgg_16_imagenet_class_trace, class_ids=range(0, 1000, 10)
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="EP",
        help="Different types of path extraction, default EP, pick between BwCU, BwAB and FwAB",
    )
    parser.add_argument(
        "--cumulative_threshold",
        type=float,
        default=0.5,
        help="cumulative threshold theta, default 0.5",
    )
    parser.add_argument(
        "--absolute_threshold",
        type=float,
        default=None,
        help="absolute threshold phi, default None",
    )
    params, unparsed = parser.parse_known_args()
    from nninst.backend.tensorflow.attack.calc_per_layer_metrics import (
        get_per_layer_metrics,
    )
    # mode.check(False)
    # mode.debug()
    mode.local()
    #mode.distributed()
    # ray_init("dell")
    ray_init()
    # ray_init("r730")
    # threshold = 0.5
    # threshold = 1
    # threshold = 0.8

    label = None

    print(f"generate class trace for label {label}")

    # save_class_traces(vgg_16_imagenet_class_trace,
    #                   np.transpose(np.reshape(np.array(range(0, 1000)), (100, 10))).flatten(),
    #                   threshold=threshold, label=label,
    #                   example_num=100, example_upperbound=1000,
    #                   merge_fn=merge_simple_trace)

    # save_vgg_16_imagenet_class_traces_low_latency(
    #     np.transpose(np.reshape(np.array(range(0, 1000)), (100, 10))).flatten(),
    #     threshold=threshold, label=label,
    #     example_num=100, example_upperbound=1000,
    #     merge_fn=merge_simple_trace)

    # check_class_traces(vgg_16_imagenet_class_trace, range(0, 1000), threshold, label, compress=True)

    # save_class_traces(vgg_16_imagenet_class_trace_compact, range(0, 1000), threshold=threshold, label=label)
    #
    # vgg_16_imagenet_trace(threshold=threshold, label=label, class_ids=range(0, 1000)).save()
    #
    # vgg_16_imagenet_self_similarity(threshold, label).save()

    # trace = vgg_16_imagenet_trace(threshold=threshold, label=label).load()
    # layers = VGG16.graph().load().layers()
    # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    #     density_per_layer = calc_density_compact_per_layer(trace, layers, key)
    #     density_per_layer.to_csv(abspath(f"vgg_16_imagenet_trace_per_layer.{key}.csv"))

    trace = vgg_16_imagenet_trace(params.cumulative_threshold, label).load()
    for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
        print(f"{key}: {calc_density_compact(trace, key)}")
