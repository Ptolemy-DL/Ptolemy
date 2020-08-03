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
import itertools
from functools import partial
import argparse
import sys

from nninst import mode
from nninst.backend.tensorflow.dataset.config import (
    CIFAR100_TEST,
    CIFAR100_TRAIN,
    IMAGENET_RAW_TRAIN,
    IMAGENET_TRAIN,
)
from nninst.backend.tensorflow.model.config import RESNET_18_CIFAR100, RESNET_50
from nninst.backend.tensorflow.model.resnet_18_cifar100 import ResNet18Cifar100
from nninst.backend.tensorflow.trace.common import (
    class_trace,
    class_trace_compact,
    class_trace_growth,
    example_trace,
    full_trace,
    save_class_traces,
    save_class_traces_low_latency,
    save_class_traces_v2,
    save_full_trace_growth,
    self_similarity,
    self_similarity_per_layer,
)
from nninst.op import Conv2dOp, DenseOp
from nninst.trace import (
    get_hybrid_backward_trace,
    get_per_input_unstructured_trace,
    get_per_receptive_field_unstructured_trace,
    get_trace,
    get_type2_trace,
    get_type3_trace,
    get_type4_trace,
    get_type7_trace,
    get_unstructured_trace,
)
from nninst.utils.ray import ray_init

__all__ = ["resnet_18_cifar100_class_trace", "resnet_18_cifar100_self_similarity"]

name = "resnet_18_cifar100"

resnet_18_cifar100_class_trace = class_trace(
    name=name, model_config=RESNET_18_CIFAR100, data_config=CIFAR100_TRAIN
)

resnet_18_cifar100_example_trace = example_trace(
    model_config=RESNET_18_CIFAR100, data_config=CIFAR100_TRAIN
)

resnet_18_cifar100_test_example_trace = example_trace(
    model_config=RESNET_18_CIFAR100, data_config=CIFAR100_TEST
)

resnet_18_cifar100_class_trace_growth = class_trace_growth(
    name=name, model_config=RESNET_18_CIFAR100, data_config=CIFAR100_TRAIN
)

resnet_18_cifar100_class_trace_compact = class_trace_compact(
    resnet_18_cifar100_class_trace, name=name, model_config=RESNET_18_CIFAR100
)

save_resnet_18_cifar100_class_traces_low_latency = save_class_traces_low_latency(
    name=name, model_config=RESNET_18_CIFAR100, data_config=CIFAR100_TRAIN
)

resnet_18_cifar100_trace = full_trace(
    name=name, class_trace_fn=resnet_18_cifar100_class_trace
)

save_resnet_18_cifar100_trace_growth = save_full_trace_growth(
    name=name, class_trace_fn=resnet_18_cifar100_class_trace
)

resnet_18_cifar100_self_similarity = self_similarity(
    name=name, trace_fn=resnet_18_cifar100_class_trace, class_ids=range(0, 100)
)

resnet_18_cifar100_self_similarity_per_layer = self_similarity_per_layer(
    name=name, trace_fn=resnet_18_cifar100_class_trace, class_ids=range(0, 100)
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
    # ray_init("gpu")
    # ray_init()
    ray_init(num_cpus=16)

    label = None
    # label = "train_50"
    # label = "train_start"
    # label = "train_start_more"

    for threshold in [
        #1.0,
        #0.9,
        #0.7,
        # 0.5,
        #0.3,
        #0.1,
        params.cumulative_threshold,
    ]:
        per_layer_metrics = lambda: get_per_layer_metrics(
            RESNET_18_CIFAR100, threshold=threshold
        )
        #hybrid_backward_traces = [
        #    [
        #        partial(
        #            get_hybrid_backward_trace,
        #            output_threshold=per_layer_metrics(),
        #            input_threshold=per_layer_metrics(),
        #            type_code=type_code,
        #        ),
        #        f"type{type_code}_trace",
        #        f"density_from_{threshold:.1f}",
        #    ]
        #    for type_code in [
                # "211111111111111111", # == type2
                #"211111111111111112",
                #"211111111111111122",
                #"211111111111111222",
                #"211111111111112222",
                #"211111111111122222",
                #"211111111111222222",
                #"211111111112222222",
                #"211111111122222222",
                #"211111111222222222",
                #"211111112222222222",
                #"211111122222222222",
                #"211111222222222222",
                #"211112222222222222",
                #"211122222222222222",
                #"211222222222222222",
                #"212222222222222222",
                #"222222222222222222",
                # "422222222222222222", # == type4
        #    ]
        #]
        if params.type == "EP":
            type_ = [get_trace, None, None]
        elif params.type == "BwCU":
            type_ = [
                partial(get_type2_trace, output_threshold=per_layer_metrics()),
                "type2_trace",
                f"density_from_{threshold:.1f}",
            ]
        elif params.type == "BwAB":
            type_ = [
                partial(
                    get_type4_trace,
                    output_threshold=per_layer_metrics(),
                    input_threshold=per_layer_metrics(),
                ),
                "type4_trace",
                f"density_from_{threshold:.1f}_absolute_{absolute_threshold:.2f}",
            ]
        elif params.type == "FwAB":
            type_ = [
                partial(
                    get_per_input_unstructured_trace,
                    output_threshold=per_layer_metrics(),
                    input_threshold=per_layer_metrics(),
                ),
                "per_input_unstructured_class_trace",
                f"density_from_{threshold:.1f}",
            ]
        else:
            print("path construction type not supported")
            sys.exit()
        for trace_info in [
            type_
            # [get_trace, None, None],
            #[
            #    partial(get_type2_trace, output_threshold=per_layer_metrics()),
            #    "type2_trace",
            #    f"density_from_{threshold:.1f}",
            #],
            # [
            #     partial(get_type3_trace, input_threshold=per_layer_metrics()),
            #     "type3_trace",
            #     f"density_from_{threshold:.1f}",
            # ],
            # [
            #     partial(
            #         get_type4_trace,
            #         output_threshold=per_layer_metrics(),
            #         input_threshold=per_layer_metrics(),
            #     ),
            #     "type4_trace",
            #     f"density_from_{threshold:.1f}",
            # ],
            # [
            #     partial(get_unstructured_trace, density=per_layer_metrics()),
            #     "unstructured_class_trace",
            #     f"density_from_{threshold:.1f}",
            # ],
            # [
            #     partial(
            #         get_per_receptive_field_unstructured_trace,
            #         output_threshold=per_layer_metrics(),
            #     ),
            #     "per_receptive_field_unstructured_class_trace",
            #     f"density_from_{threshold:.1f}",
            # ],
            # [
            #     partial(
            #         get_type7_trace,
            #         density=per_layer_metrics(),
            #         input_threshold=per_layer_metrics(),
            #     ),
            #     "type7_trace",
            #     f"density_from_{threshold:.1f}",
            # ],
            # [
            #     partial(
            #         get_per_input_unstructured_trace,
            #         output_threshold=per_layer_metrics(),
            #         input_threshold=per_layer_metrics(),
            #     ),
            #     "per_input_unstructured_class_trace",
            #     f"density_from_{threshold:.1f}",
            # ],
            # *hybrid_backward_traces,
        ]:
            trace_fn, trace_type, trace_parameter = trace_info
            save_class_traces_v2(
                partial(
                    resnet_18_cifar100_class_trace,
                    trace_fn=trace_fn,
                    trace_type=trace_type,
                    trace_parameter=trace_parameter,
                    threshold=threshold,
                    label=label,
                    example_num=500,
                    example_upperbound=500,
                    # cache=False,
                ),
                range(0, 100),
            )

            save_class_traces_v2(
                partial(
                    resnet_18_cifar100_class_trace_compact,
                    trace_type=trace_type,
                    trace_parameter=trace_parameter,
                    threshold=threshold,
                    label=label,
                    # cache=False,
                ),
                range(0, 100),
            )

            # resnet_18_cifar100_self_similarity(threshold=threshold, label=label).save()

        # class_id = 0
        # image_id = 0
        # resnet_18_cifar100_example_trace(class_id, image_id, threshold, label)

        # graph = ResNet18Cifar100.graph().load()
        # layers = graph.ops_in_layers(Conv2dOp, DenseOp)
        #
        # for layer_id in [-2, -3, -4]:
        #     resnet_18_cifar100_self_similarity_per_layer(
        #         layer_name=layers[layer_id], threshold=threshold, label=label
        #     ).save()
