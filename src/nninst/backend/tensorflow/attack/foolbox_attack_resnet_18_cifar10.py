import itertools
import os
from functools import partial

import numpy as np
import tensorflow as tf
from foolbox.attacks import (
    FGSM,
    DeepFoolAttack,
    IterativeGradientSignAttack,
    SaliencyMapAttack,
)

from nninst import mode
from nninst.backend.tensorflow.attack.calc_density import trace_density
from nninst.backend.tensorflow.attack.calc_per_layer_metrics import (
    get_per_layer_metrics,
    trace_per_layer_metrics,
)
from nninst.backend.tensorflow.attack.common import (
    resnet_18_cifar10_real_metrics_per_layer,
)
from nninst.backend.tensorflow.attack.cw_attack import cw_generate_adversarial_example
from nninst.backend.tensorflow.attack.cw_attacks import CarliniL2
from nninst.backend.tensorflow.attack.foolbox_attack import (
    foolbox_generate_adversarial_example,
    random_targeted,
)
from nninst.backend.tensorflow.attack.foolbox_attacks.fgsm import (
    TargetedFGSM,
    TargetedIterativeFGSM,
)
from nninst.backend.tensorflow.attack.random_attack import RandomAttack
from nninst.backend.tensorflow.dataset import imagenet
from nninst.backend.tensorflow.model.config import RESNET_18_CIFAR10
from nninst.backend.tensorflow.model.resnet_18_cifar10 import ResNet18Cifar10
from nninst.backend.tensorflow.trace.resnet_18_cifar10_class_trace_v2 import (
    resnet_18_cifar10_class_trace_compact,
)
from nninst.backend.tensorflow.trace.utils import (
    can_support_diff,
    get_entry_points,
    get_select_seed_fn,
    get_variant,
)
from nninst.channel_trace import get_channel_trace
from nninst.statistics import calc_trace_side_overlap_both_compact
from nninst.trace import (
    TraceKey,
    density_name,
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
from nninst.utils.alternative import alt, alts
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


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
    # mode.debug()
    # mode.distributed()
    mode.local()
    # ray_init("dell")
    ray_init()
    attacks = {
        "normal": [None],
        "FGSM": [FGSM],
        "FGSM_targeted": [random_targeted(TargetedFGSM, 1, 999)],
        "FGSM_iterative_targeted": [random_targeted(TargetedIterativeFGSM, 1, 999)],
        "BIM": [IterativeGradientSignAttack],
        "JSMA": [SaliencyMapAttack],
        "DeepFool": [DeepFoolAttack],
        "DeepFool_full": [DeepFoolAttack, dict(subsample=None)],
        "CWL2": [CarliniL2],
        "CWL2_confidence=3.5": [partial(CarliniL2, confidence=3.5)],
        "CWL2_confidence=14": [partial(CarliniL2, confidence=14)],
        "CWL2_confidence=28": [partial(CarliniL2, confidence=28)],
        "CWL2_target=500": [CarliniL2],
        "CWL2_confidence=28_target=500": [partial(CarliniL2, confidence=28)],
        "patch": [None],
        "patch_scale=0.1": [None],
        "patch_scale=0.2": [None],
        "patch_scale=0.3": [None],
        "patch_scale=0.4": [None],
        "patch_scale=0.5": [None],
        "new_patch_scale=0.1": [None],
        "new_patch_scale=0.2": [None],
        "new_patch_scale=0.3": [None],
        "new_patch_scale=0.4": [None],
        "new_patch_scale=0.5": [None],
        "negative_example": [None],
        "negative_example_top5": [None],
        "negative_example_out_of_top5": [None],
        "Random": [RandomAttack],
    }

    label = None
    variant = None
    use_weight = False
    # use_weight = True
    print(f"attack model with label {label} using Foolbox")
    attack_name, generate_adversarial_fn = alts(
        ["normal", None],
        # ["DeepFool", foolbox_generate_adversarial_example],
        # ["FGSM", foolbox_generate_adversarial_example],
        # ["FGSM_targeted", foolbox_generate_adversarial_example],
        # ["FGSM_iterative_targeted", foolbox_generate_adversarial_example],
        # ["BIM", foolbox_generate_adversarial_example],
        # ["JSMA", foolbox_generate_adversarial_example],
        ["CWL2", cw_generate_adversarial_example],
        # ["CWL2_confidence=3.5", cw_generate_adversarial_example],
        # ["CWL2_confidence=14", cw_generate_adversarial_example],
        # ["CWL2_confidence=28", cw_generate_adversarial_example],
        # ["CWL2_target=500", cw_generate_adversarial_example],
        # ["CWL2_confidence=28_target=500", cw_generate_adversarial_example],
        # ["CWL2_confidence=28_target=500", cw_generate_adversarial_example],
        # ["patch", patch_generate_adversarial_example],
        # ["patch_scale=0.1", patch_generate_adversarial_example],
        # ["patch_scale=0.2", patch_generate_adversarial_example],
        # ["patch_scale=0.3", patch_generate_adversarial_example],
        # ["patch_scale=0.4", patch_generate_adversarial_example],
        # ["patch_scale=0.5", patch_generate_adversarial_example],
        # ["new_patch_scale=0.1", patch_generate_adversarial_example],
        # ["new_patch_scale=0.2", patch_generate_adversarial_example],
        # ["new_patch_scale=0.3", patch_generate_adversarial_example],
        # ["new_patch_scale=0.4", patch_generate_adversarial_example],
        # ["new_patch_scale=0.5", patch_generate_adversarial_example],
        # ["negative_example", None],
        # ["negative_example_top5", None],
        # ["negative_example_out_of_top5", None],
        # ["Random", foolbox_generate_adversarial_example],
    )
    per_layer_metrics = lambda: get_per_layer_metrics(RESNET_18_CIFAR10, threshold=params.cumulative_threshold)
    #hybrid_backward_traces = [
    #    [
    #        partial(
    #            get_hybrid_backward_trace,
    #            output_threshold=per_layer_metrics(),
    #            input_threshold=per_layer_metrics(),
    #            type_code=type_code,
    #        ),
    #        f"type{type_code}_density_from_0.5",
    #        f"type{type_code}_trace",
    #        "density_from_0.5",
    #    ]
    #    for type_code in [
            # "211111111111111111", # == type2
    #        "211111111111111112",
    #        "211111111111111122",
    #        "211111111111111222",
    #        "211111111111112222",
    #        "211111111111122222",
    #        "211111111111222222",
    #        "211111111112222222",
    #        "211111111122222222",
    #        "211111111222222222",
    #        "211111112222222222",
    #        "211111122222222222",
    #        "211111222222222222",
    #        "211112222222222222",
    #        "211122222222222222",
    #        "211222222222222222",
    #        "212222222222222222",
    #        "222222222222222222",
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
    trace_fn, trace_label, trace_type, trace_parameter = alts(
        type_
        # [get_trace, None, None, None],
        # [get_channel_trace, "per_channel", "class_channel_trace", None],
        #[
        #    partial(get_type2_trace, output_threshold=per_layer_metrics()),
        #    "type2_density_from_0.5",
        #    "type2_trace",
        #    "density_from_0.5",
        #],
        # [
        #     partial(get_type3_trace, input_threshold=per_layer_metrics()),
        #     "type3_density_from_0.5",
        #     "type3_trace",
        #     "density_from_0.5",
        # ],
        # [
        #     partial(
        #         get_type4_trace,
        #         output_threshold=per_layer_metrics(),
        #         input_threshold=per_layer_metrics(),
        #     ),
        #     "type4_density_from_0.5",
        #     "type4_trace",
        #     "density_from_0.5",
        # ],
        # [
        #     partial(get_unstructured_trace, density=per_layer_metrics()),
        #     "unstructured_density_from_0.5",
        #     "unstructured_class_trace",
        #     "density_from_0.5",
        # ],
        #[
        #    partial(
        #        get_per_receptive_field_unstructured_trace,
        #        output_threshold=per_layer_metrics(),
        #    ),
        #    "per_receptive_field_unstructured_density_from_0.5",
        #    "per_receptive_field_unstructured_class_trace",
        #    "density_from_0.5",
        #],
        # [
        #     partial(
        #         get_type7_trace,
        #         density=per_layer_metrics(),
        #         input_threshold=per_layer_metrics(),
        #     ),
        #     "type7_density_from_0.5",
        #     "type7_trace",
        #     "density_from_0.5",
        # ],
        # [
        #     partial(
        #         get_per_input_unstructured_trace,
        #         output_threshold=per_layer_metrics(),
        #         input_threshold=per_layer_metrics(),
        #     ),
        #     "per_input_unstructured_density_from_0.5",
        #     "per_input_unstructured_class_trace",
        #     "density_from_0.5",
        # ],
        # *hybrid_backward_traces
    )
    threshold = alt(
        # 1.0,
        # 0.9,
        # 0.7,
        0.5,
        # 0.3,
        # 0.1,
    )
    example_num = alt(
        # 100,
        # 400,
        # 700,
        # 1000,
        0
    )
    layer_num = alt(
        0,
        # 3,
        # 4,
    )
    seed_threshold = alt(
        None,
        # 0.5,
        # 0.1,
        # 0.01,
        # 0.001,
    )
    per_channel = alt(
        # True,
        False
    )
    rank = alt(
        # None,
        1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
        # 8,
        # 9,
        # 10,
    )

    use_point = alt(
        # True,
        False
    )
    compare_with_full = alt(
        False,
        # True,
    )
    get_overlap_with_all_class = alt(
        True,
        # False,
    )

    for config in (
        (attack_name | generate_adversarial_fn)
        * (trace_fn | trace_label | trace_type | trace_parameter)
        * threshold
        * example_num
        * layer_num
        * seed_threshold
        * per_channel
        * rank
        * use_point
        * compare_with_full
        * get_overlap_with_all_class
    ):
        with config:
            print(f"config: {list(config.values())}")
            entry_points = get_entry_points(
                ResNet18Cifar10.graph().load(), layer_num.value
            )
            select_seed_fn = get_select_seed_fn(seed_threshold.value)
            support_diff = can_support_diff(layer_num.value)
            variant = get_variant(
                example_num.value, layer_num.value, seed_threshold.value
            )

            if variant is not None:
                label_name = f"{label}_{variant}"
            else:
                label_name = label

            if use_weight:
                label_name = f"{label_name}_weight"
            elif use_point.value:
                label_name = f"{label_name}_point"
            if compare_with_full.value:
                label_name = f"{label_name}_vs_full"
            if trace_label.value is not None:
                label_name = f"{label_name}_{trace_label.value}"
            if rank.value is not None:
                label_name = f"{label_name}_rank{rank.value}"
            if get_overlap_with_all_class.value:
                label_name = f"{label_name}_all_class"

            path_template = (
                "resnet_18_cifar10_real_metrics_per_layer_{0:.1f}_{1}_{2}.csv"
            )
            per_node = False
            resnet_18_overlap_ratio = resnet_18_cifar10_real_metrics_per_layer(
                attack_name=attack_name.value,
                attack_fn=attacks[attack_name.value][0],
                generate_adversarial_fn=generate_adversarial_fn.value,
                trace_fn=partial(
                    trace_fn.value,
                    select_fn=lambda input: arg_approx(input, threshold.value),
                    select_seed_fn=select_seed_fn,
                    entry_points=entry_points,
                ),
                class_trace_fn=lambda class_id: resnet_18_cifar10_class_trace_compact(
                    class_id,
                    threshold.value,
                    label=label,
                    variant=variant,
                    trace_type=None if compare_with_full.value else trace_type.value,
                    trace_parameter=None
                    if compare_with_full.value
                    else trace_parameter.value,
                ),
                path="metrics/"
                + path_template.format(threshold.value, attack_name.value, label_name),
                per_node=per_node,
                per_channel=per_channel.value,
                use_weight=use_weight,
                support_diff=support_diff,
                threshold=threshold.value,
                rank=rank.value,
                use_point=use_point.value,
                get_overlap_with_all_class=get_overlap_with_all_class.value,
                label=label,
                **(
                    attacks[attack_name.value][1]
                    if len(attacks[attack_name.value]) == 2
                    else {}
                ),
            )

            resnet_18_overlap_ratio.save()
