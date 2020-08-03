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
from nninst.backend.tensorflow.dataset.config import CIFAR10_TEST, CIFAR10_TRAIN
from nninst.backend.tensorflow.model.config import RESNET_18_CIFAR10
from nninst.backend.tensorflow.model.resnet_18_cifar10 import ResNet18Cifar10
from nninst.backend.tensorflow.trace.common import (
    class_trace,
    class_trace_compact,
    class_trace_growth,
    example_trace,
    full_trace,
    inter_class_similarity,
    inter_class_similarity_frequency,
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

__all__ = ["resnet_18_cifar10_class_trace", "resnet_18_cifar10_self_similarity"]

name = "resnet_18_cifar10"

resnet_18_cifar10_class_trace = class_trace(
    name=name, model_config=RESNET_18_CIFAR10, data_config=CIFAR10_TRAIN
)

resnet_18_cifar10_class_trace_growth = class_trace_growth(
    name=name, model_config=RESNET_18_CIFAR10, data_config=CIFAR10_TRAIN
)

resnet_18_cifar10_class_trace_compact = class_trace_compact(
    resnet_18_cifar10_class_trace, name=name, model_config=RESNET_18_CIFAR10
)

save_resnet_18_cifar10_class_traces_low_latency = save_class_traces_low_latency(
    name=name, model_config=RESNET_18_CIFAR10, data_config=CIFAR10_TRAIN
)

resnet_18_cifar10_trace = full_trace(
    name=name, class_trace_fn=resnet_18_cifar10_class_trace
)

save_resnet_18_cifar10_trace_growth = save_full_trace_growth(
    name=name, class_trace_fn=resnet_18_cifar10_class_trace
)

resnet_18_cifar10_self_similarity = self_similarity(
    name=name, trace_fn=resnet_18_cifar10_class_trace, class_ids=range(0, 10)
)

resnet_18_cifar10_inter_class_similarity = inter_class_similarity(
    name=name,
    trace_fn=resnet_18_cifar10_class_trace,
    class_ids=range(0, 10),
    start_index_map={"left": 0, "right": 2500},
)

resnet_18_cifar10_inter_class_similarity_frequency = inter_class_similarity_frequency(
    name=name,
    trace_fn=resnet_18_cifar10_class_trace,
    class_ids=range(0, 10),
    start_index_map={"left": 0, "right": 2500},
)

resnet_18_cifar10_self_similarity_per_layer = self_similarity_per_layer(
    name=name, trace_fn=resnet_18_cifar10_class_trace, class_ids=range(0, 10)
)

if __name__ == "__main__":
    from nninst.backend.tensorflow.attack.calc_per_layer_metrics import (
        get_per_layer_metrics,
    )

    # mode.check(False)
    # mode.debug()
    mode.local()
    # mode.distributed()
    # ray_init("dell")
    # ray_init("gpu")
    # ray_init()
    ray_init()

    threshold = 0.5
    # threshold = 1
    # threshold = 0.8

    label = None
    # label = "train_50"
    # label = "train_start"
    # label = "train_start_more"

    # frequency = int(2500 * 0.05)
    frequency = int(2500 * 0.1)
    # frequency = int(2500 * 0.3)
    # frequency = int(2500 * 0.5)
    # frequency = int(2500 * 0.7)
    # frequency = int(2500 * 0.8)
    # frequency = int(2500 * 0.9)

    per_layer_metrics = lambda: get_per_layer_metrics(RESNET_18_CIFAR10, threshold=0.5)
    hybrid_backward_traces = [
        [
            partial(
                get_hybrid_backward_trace,
                output_threshold=per_layer_metrics(),
                input_threshold=per_layer_metrics(),
                type_code=type_code,
            ),
            f"type{type_code}_trace",
            "density_from_0.5",
        ]
        for type_code in [
            # "211111111111111111", # == type2
            "211111111111111112",
            "211111111111111122",
            "211111111111111222",
            "211111111111112222",
            "211111111111122222",
            "211111111111222222",
            "211111111112222222",
            "211111111122222222",
            "211111111222222222",
            "211111112222222222",
            "211111122222222222",
            "211111222222222222",
            "211112222222222222",
            "211122222222222222",
            "211222222222222222",
            "212222222222222222",
            "222222222222222222",
            # "422222222222222222", # == type4
        ]
    ]
    for trace_info in [
        [get_trace, None, None],
        # [
        #     partial(get_type2_trace, output_threshold=per_layer_metrics()),
        #     "type2_trace",
        #     "density_from_0.5",
        # ],
        # [
        #     partial(get_type3_trace, input_threshold=per_layer_metrics()),
        #     "type3_trace",
        #     "density_from_0.5",
        # ],
        # [
        #     partial(
        #         get_type4_trace,
        #         output_threshold=per_layer_metrics(),
        #         input_threshold=per_layer_metrics(),
        #     ),
        #     "type4_trace",
        #     "density_from_0.5",
        # ],
        # [
        #     partial(get_unstructured_trace, density=per_layer_metrics()),
        #     "unstructured_class_trace",
        #     "density_from_0.5",
        # ],
        # [
        #     partial(
        #         get_per_receptive_field_unstructured_trace,
        #         output_threshold=per_layer_metrics(),
        #     ),
        #     "per_receptive_field_unstructured_class_trace",
        #     "density_from_0.5",
        # ],
        # [
        #     partial(
        #         get_type7_trace,
        #         density=per_layer_metrics(),
        #         input_threshold=per_layer_metrics(),
        #     ),
        #     "type7_trace",
        #     "density_from_0.5",
        # ],
        # [
        #     partial(
        #         get_per_input_unstructured_trace,
        #         output_threshold=per_layer_metrics(),
        #         input_threshold=per_layer_metrics(),
        #     ),
        #     "per_input_unstructured_class_trace",
        #     "density_from_0.5",
        # ],
        # *hybrid_backward_traces
    ]:
        trace_fn, trace_type, trace_parameter = trace_info
        # for example_start_index in [0, 2500]:
        #     variant_for_trace = f"[start_index={example_start_index}]"
        #     save_class_traces_v2(
        #         partial(
        #             resnet_18_cifar10_class_trace,
        #             trace_fn=trace_fn,
        #             trace_type=trace_type,
        #             trace_parameter=trace_parameter,
        #             threshold=threshold,
        #             label=label,
        #             variant=variant_for_trace,
        #             example_start_index=example_start_index,
        #             example_num=2500,
        #             example_upperbound=2500,
        #             use_map_reduce=True,
        #             reduce_batch_size=200,
        #             # reduce_batch_size=320,
        #             # cache=False,
        #         ),
        #         range(0, 10),
        #     )
        #
        #     save_class_traces_v2(
        #         partial(
        #             resnet_18_cifar10_class_trace_compact,
        #             trace_type=trace_type,
        #             trace_parameter=trace_parameter,
        #             threshold=threshold,
        #             label=label,
        #             variant=variant_for_trace,
        #             # cache=False,
        #         ),
        #         range(0, 10),
        #     )

        for layer_name in [
            None,
            *ResNet18Cifar10.graph().load().ops_in_layers(Conv2dOp, DenseOp),
        ]:
            # resnet_18_cifar10_inter_class_similarity(threshold=threshold, label=label, layer_name=layer_name).save()
            resnet_18_cifar10_inter_class_similarity_frequency(
                threshold=threshold,
                frequency=frequency,
                label=label,
                layer_name=layer_name,
            ).save()

    # class_id = 0
    # image_id = 0
    # resnet_18_cifar10_example_trace(class_id, image_id, threshold, label)

    # graph = ResNet18Cifar10.graph().load()
    # layers = graph.ops_in_layers(Conv2dOp, DenseOp)
    #
    # for layer_id in [-2, -3, -4]:
    #     resnet_18_cifar10_self_similarity_per_layer(
    #         layer_name=layers[layer_id], threshold=threshold, label=label
    #     ).save()
