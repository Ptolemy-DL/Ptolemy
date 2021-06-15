import argparse
import itertools
import os
from functools import partial

from foolbox.attacks import (
    FGSM,
    DeepFoolAttack,
    IterativeGradientSignAttack,
    SaliencyMapAttack,
)

from nninst import mode
from nninst.backend.tensorflow.attack.calc_per_layer_metrics import (
    get_per_layer_metrics,
)
from nninst.backend.tensorflow.attack.common import (
    alexnet_imagenet_real_metrics_per_layer,
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
from nninst.backend.tensorflow.attack.utils import parse_path_generation_args
from nninst.backend.tensorflow.dataset import imagenet
from nninst.backend.tensorflow.model import AlexNet
from nninst.backend.tensorflow.model.config import ALEXNET
from nninst.backend.tensorflow.trace.alexnet_imagenet_class_trace import (
    alexnet_imagenet_class_trace_compact,
)
from nninst.backend.tensorflow.trace.utils import (
    can_support_diff,
    get_entry_points,
    get_select_seed_fn,
    get_variant,
)
from nninst.statistics import calc_trace_side_overlap_both_compact
from nninst.trace import (
    early_stop_hook,
    get_hybrid_backward_trace,
    get_per_input_unstructured_trace,
    get_trace,
    get_type2_trace,
    get_type4_trace,
)
from nninst.utils.alternative import alt, alts
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

if __name__ == "__main__":
    absolute_threshold, cumulative_threshold, type_ = parse_path_generation_args(ALEXNET)
    # mode.debug()
    # mode.distributed()
    mode.local()
    # ray_init("dell")
    ray_init()

    label = "import"
    # label = "without_dropout"
    # label = "import_old"
    # label = "best_in_10"
    # label = "worst_in_10"
    # label = "import"
    # label = "norm"
    variant = None
    # variant = "compare_with_full"
    # variant = "intersect"
    use_weight = False
    # use_weight = True
    print(f"attack model with label {label} using Foolbox")
    attack_name = alt(
        "normal",
        "DeepFool",
        # "Adaptive_layer1",
        # "Adaptive_layer2",
        # "Adaptive_layer3",
        # "Adaptive_layer4",
        # "Adaptive_layer5",
        # "Adaptive_layer6",
        # "Adaptive_layer7",
        # "Adaptive_layer8",
        # "Adaptive_layer9",
        # "Adaptive_cos_layer9",
        # "Adaptive_layer4",
        # "Adaptive_return_late",
        # "Adaptive_random_start",
        # "Adaptive_iterations_400",
        # "Adaptive_layer4_iterations_400",
        "FGSM",
        # "FGSM_targeted",
        # "FGSM_iterative_targeted",
        "BIM",
        "JSMA",
        "CWL2",
        # "CWL2_confidence=3.5",
        # "CWL2_confidence=14",
        # "CWL2_confidence=28",
        # "CWL2_target=500",
        # "CWL2_confidence=28_target=500",
        # "CWL2_confidence=28_target=500",
        # "patch",
        # "patch_scale=0.1",
        # "patch_scale=0.2",
        # "patch_scale=0.3",
        # "patch_scale=0.4",
        # "patch_scale=0.5",
        # "new_patch_scale=0.1",
        # "new_patch_scale=0.2",
        # "new_patch_scale=0.3",
        # "new_patch_scale=0.4",
        # "new_patch_scale=0.5",
        # "negative_example",
        # "negative_example_top5",
        # "negative_example_out_of_top5",
        # "Random",
    )
    topk_share_range = alt(
        # 2,
        # 3,
        # 5,
        # 6,
        # 7,
        # 8,
        9,
        # 10,
        # 20,
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
        2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
        # 8,
        # 9,
        # 10,
    )
    early_stop_layer_num = alt(
        None,
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
    for threshold, absolute_threshold in itertools.product(
        [
            # 1.0,
            # 0.9,
            # 0.7,
            #0.5,
            # 0.3,
            # 0.1,
            cumulative_threshold,
        ],
        [
            absolute_threshold,
            #None,
            # 0.05,
            # 0.1,
            # 0.2,
            # 0.3,
            # 0.4,
        ],
    ):
        # hybrid_backward_traces = [
        #     [
        #         partial(
        #             get_hybrid_backward_trace,
        #             output_threshold=per_layer_metrics(),
        #             input_threshold=per_layer_metrics(),
        #             type_code=type_code,
        #         ),
        #         f"type{type_code}_density_from_{threshold:.1f}",
        #         f"type{type_code}_trace",
        #         f"density_from_{threshold:.1f}",
        #     ]
        #     for type_code in [
        #         # "21111111", # == type2
        #         "21111112",
        #         "21111122",
        #         "21111222",
        #         "21112222",
        #         "21122222",
        #         "21222222",
        #         "22222222",
        #         # "42222222", # == type4
        #     ]
        # ]
        trace_fn, trace_label, trace_type, trace_parameter = alts(
            type_
            #[get_trace, None, None, None],  # type1
            # [get_channel_trace, "per_channel", "class_channel_trace", None],
            # [
            #     partial(get_type2_trace, output_threshold=per_layer_metrics()),
            #     f"type2_density_from_{threshold:.1f}",
            #     "type2_trace",
            #     f"density_from_{threshold:.1f}",
            # ],
            # [
            #     partial(get_type3_trace, input_threshold=per_layer_metrics()),
            #     f"type3_density_from_{threshold:.1f}",
            #     "type3_trace",
            #     f"density_from_{threshold:.1f}",
            # ],
            # [
            #     partial(
            #         get_type4_trace,
            #         output_threshold=per_layer_metrics(),
            #         input_threshold=per_layer_metrics(),
            #     ),
            #     f"type4_density_from_{threshold:.1f}",
            #     "type4_trace",
            #     f"density_from_{threshold:.1f}",
            # ],
            # [
            #     partial(
            #         get_type4_trace,
            #         output_threshold=per_layer_metrics(),
            #         input_threshold=per_layer_metrics(),
            #     ),
            #     f"type4_density_from_{threshold:.1f}_absolute_{absolute_threshold:.2f}",
            #     "type4_trace",
            #     f"density_from_{threshold:.1f}_absolute_{absolute_threshold:.2f}",
            # ],
            # [
            #     partial(get_unstructured_trace, density=per_layer_metrics()),
            #     f"unstructured_density_from_{threshold:.1f}",
            #     "unstructured_class_trace",
            #     f"density_from_{threshold:.1f}",
            # ], # type5
            # [
            #     partial(
            #         get_per_receptive_field_unstructured_trace,
            #         output_threshold=per_layer_metrics(),
            #     ),
            #     f"per_receptive_field_unstructured_density_from_{threshold:.1f}",
            #     "per_receptive_field_unstructured_class_trace",
            #     f"density_from_{threshold:.1f}",
            # ], # type6
            # [
            #     partial(
            #         get_type7_trace,
            #         density=per_layer_metrics(),
            #         input_threshold=per_layer_metrics(),
            #     ),
            #     f"type7_density_from_{threshold:.1f}",
            #     "type7_trace",
            #     f"density_from_{threshold:.1f}",
            # ],
            # [
            #     partial(
            #         get_per_input_unstructured_trace,
            #         output_threshold=per_layer_metrics(),
            #         input_threshold=per_layer_metrics(),
            #     ),
            #     f"per_input_unstructured_density_from_{threshold:.1f}",
            #     "per_input_unstructured_class_trace",
            #     f"density_from_{threshold:.1f}",
            # ], # type8
            # *hybrid_backward_traces
        )
        for config in (
            attack_name
            * (trace_fn | trace_label | trace_type | trace_parameter)
            * topk_share_range
            * example_num
            * layer_num
            * seed_threshold
            * per_channel
            * rank
            * use_point
            * compare_with_full
            * early_stop_layer_num
        ):
            with config:
                print(f"config: {list(config.values())}")
                topk_calc_range = 2
                entry_points = get_entry_points(AlexNet.graph().load(), layer_num.value)
                select_seed_fn = get_select_seed_fn(seed_threshold.value)
                support_diff = can_support_diff(layer_num.value)
                variant = get_variant(
                    example_num=example_num.value,
                    layer_num=layer_num.value,
                    seed_threshold=seed_threshold.value,
                    early_stop_layer_num=early_stop_layer_num.value,
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
                # if per_channel.value:
                #     label_name = f"{label_name}_per_channel"
                # if is_unstructured.value:
                #     label_name = (
                #         f"{label_name}_unstructured_density_{density_name(density.value)}"
                #     )
                if trace_label.value is not None:
                    label_name = f"{label_name}_{trace_label.value}"
                if rank.value is not None:
                    label_name = f"{label_name}_rank{rank.value}"

                # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio(
                #     attack_fn=attacks[attack_name][0],
                #     generate_adversarial_fn=generate_adversarial_example,
                #     class_trace_fn=lambda class_id: alexnet_imagenet_class_trace_compact(class_id, threshold, label=label),
                #     # class_trace_fn=lambda class_id: lenet_mnist_class_trace(class_id, threshold),
                #     select_fn=lambda input: arg_approx(input, threshold),
                #     overlap_fn=calc_trace_side_overlap_compact,
                #     # overlap_fn=calc_iou,
                #     # overlap_fn=calc_class_trace_side_overlap,
                #     # overlap_fn=calc_class_trace_side_overlap_norm,
                #     # overlap_fn=calc_weighted_iou,
                #     path='alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.csv'.format(
                #         threshold, attack_name, label),
                #     # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.iou.csv'.format(threshold, attack_name, label),
                #     # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.class_side.csv'.format(
                #     # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.wo_pool.csv'.format(
                #     # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.class_side_norm.csv'.format(
                #     # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.weighted_iou.csv'.format(
                #     #     threshold, attack_name, label),
                #     preprocessing=(_CHANNEL_MEANS, 1),
                #     **(attacks[attack_name][1] if len(attacks[attack_name]) == 2 else {}),
                # )
                # alexnet_overlap_ratio.save()

                # print("edge:")
                # summary = get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.EDGE)
                # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.csv".format(
                #     threshold=threshold, label=label)
                # file_exists = os.path.exists(summary_file)
                # with open(summary_file, "a") as csv_file:
                #     headers = ["attack"] + list(summary.keys())
                #     writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n', fieldnames=headers)
                #     if not file_exists:
                #         writer.writeheader()
                #     writer.writerow({"attack": attack_name, **summary})
                # print(summary)
                # print("weight:")
                # print(get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.WEIGHT))
                # print("point:")
                # print(get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.POINT))

                # for overlay_threshold in np.arange(0, 1.01, 0.01):
                #     # summary = get_overlay_summary(alexnet_overlap_ratio.load(), TraceKey.EDGE, overlay_threshold)
                #     summary = get_overlay_summary(alexnet_overlap_ratio.load(), TraceKey.WEIGHT, overlay_threshold)
                #     summary_file = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}.csv".format(
                #         # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.iou.csv".format(
                #         # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.class_side.csv".format(
                #         # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.wo_pool.csv".format(
                #         # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.class_side_norm.csv".format(
                #         # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.weighted_iou.csv".format(
                #         threshold=threshold, attack=attack_name, label=label)
                #     file_exists = os.path.exists(summary_file)
                #     with open(summary_file, "a") as csv_file:
                #         headers = ["attack"] + list(summary.keys())
                #         writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n', fieldnames=headers)
                #         if not file_exists:
                #             writer.writeheader()
                #         writer.writerow({"attack": attack_name, **summary})

                # overlap_fn = calc_trace_side_overlap_compact
                overlap_fn = calc_trace_side_overlap_both_compact
                # overlap_fn = calc_weighted_iou
                # overlap_fn = calc_class_trace_side_overlap_compact
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_{0:.1f}_{1}_{2}_top5_diff.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_{0:.1f}_{1}_{2}_top5_diff_same_class_trace.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_{0:.1f}_{1}_{2}_top5_diff_all.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top5_diff_all.foolbox.csv"
                # path_template = ("alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top"
                #                  + str(topk_share_range)
                #                  + "_diff_all_uint8.foolbox.csv")
                # path_template = ("alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top"
                #                  + str(topk_share_range)
                #                  + "_diff_all.foolbox.csv")
                # path_template = ("alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top"
                #                  + str(topk_share_range)
                #                  + "_logit_diff.foolbox.csv")
                # path_template = "alexnet_imagenet_ideal_metrics_{0:.1f}_{1}_{2}.csv"
                # path_template = "alexnet_imagenet_fc_layer_path_ideal_metrics_{0:.1f}_{1}_{2}.csv"
                # path_template = "alexnet_imagenet_ideal_metrics_per_layer_{0:.1f}_{1}_{2}.csv"
                # path_template = "alexnet_imagenet_real_metrics_per_layer_{0:.1f}_{1}_{2}.csv"
                path_template = (
                    "alexnet_imagenet_real_metrics_per_layer_{0:.1f}_{1}_{2}.csv"
                )
                # path_template = "alexnet_imagenet_real_metrics_per_layer_targeted0_{0:.1f}_{1}_{2}.csv"
                # path_template = "alexnet_imagenet_real_metrics_per_layer_targeted500_{0:.1f}_{1}_{2}.csv"
                # path_template = "alexnet_imagenet_real_metrics_per_layer_targeted800_{0:.1f}_{1}_{2}.csv"
                # path_template = "alexnet_imagenet_real_metrics_per_layer_{0:.1f}_{1}_{2}_flip_diff.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top2_diff_all.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top5_unique.foolbox.csv"
                # path_template = "alexnet_imagenet_train_class_overlap_ratio_{0:.1f}_{1}_{2}_top5_unique.weight.foolbox.csv"
                # path_template = "alexnet_imagenet_train_class_overlap_ratio_{0:.1f}_{1}_{2}_top5_unique.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_{0:.1f}_{1}_{2}_top5_diff_all_online.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_per_node_{0:.1f}_{1}_{2}_top5_diff.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_{0:.1f}_{1}_{2}_top5_diff_train.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_{0:.1f}_{1}_{2}_weighted_iou.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_{0:.1f}_{1}_{2}_weighted_iou_class_0.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_per_node_{0:.1f}_{1}_{2}_weighted_iou_class_0.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_per_node_{0:.1f}_{1}_{2}_weighted_iou_class_0.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_per_node_{0:.1f}_{1}_{2}_weighted_iou.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_per_node_{0:.1f}_{1}_{2}_weighted_iou.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_per_node_{0:.1f}_{1}_{2}.foolbox.csv"
                # path_template = "alexnet_imagenet_class_channel_overlap_ratio_{0:.1f}_{1}_{2}_class_0.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_full.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_train_in_trace.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_train_not_merged.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top5.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top5_all.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_error.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_rand.foolbox.csv"
                # path_template = "alexnet_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top5_rand.foolbox.csv"
                per_node = False
                # per_node = True
                # per_channel = True
                # per_channel = False
                # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio(
                # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_top5_diff_uint8(
                # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_top5_diff(
                # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_logit_diff(
                # alexnet_overlap_ratio = alexnet_imagenet_ideal_metrics(
                # alexnet_overlap_ratio = alexnet_imagenet_fc_layer_path_ideal_metrics(
                # alexnet_overlap_ratio = alexnet_imagenet_negative_example_ideal_metrics_per_layer(
                # alexnet_overlap_ratio = alexnet_imagenet_ideal_metrics_per_layer(
                alexnet_overlap_ratio = alexnet_imagenet_real_metrics_per_layer(
                    # alexnet_overlap_ratio = alexnet_imagenet_real_metrics_per_layer_targeted(target_class=0)(
                    # alexnet_overlap_ratio = alexnet_imagenet_real_metrics_per_layer_targeted(target_class=500)(
                    # alexnet_overlap_ratio = alexnet_imagenet_real_metrics_per_layer_targeted(target_class=800)(
                    # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_top5_unique(
                    attack_name=attack_name.value,
                    # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_top5(
                    trace_fn=partial(
                        trace_fn.value,
                        select_fn=lambda input: arg_approx(input, threshold),
                        select_seed_fn=select_seed_fn,
                        entry_points=entry_points,
                        stop_hook=early_stop_hook(early_stop_layer_num.value)
                        if early_stop_layer_num.value is not None
                        else None,
                    ),
                    # class_trace_fn=lambda class_id: alexnet_imagenet_class_trace(class_id, threshold, label=label),
                    # class_trace_fn=lambda class_id: alexnet_imagenet_class_trace_compact(class_id, threshold, label=label),
                    class_trace_fn=lambda class_id: alexnet_imagenet_class_trace_compact(
                        # class_trace_fn=lambda class_id: alexnet_imagenet_class_trace_compact(
                        # class_trace_fn=lambda class_id: alexnet_imagenet_class_channel_trace(
                        # class_trace_fn=lambda class_id: alexnet_imagenet_class_unique_trace_compact(
                        # class_trace_fn=lambda class_id: alexnet_imagenet_class_trace(
                        class_id,
                        threshold,
                        label=label,
                        variant=variant,
                        trace_type=None
                        if compare_with_full.value
                        else trace_type.value,
                        trace_parameter=None
                        if compare_with_full.value
                        else trace_parameter.value,
                    ),
                    # class_trace_fn=lambda class_id: lenet_mnist_class_trace(class_id, threshold),
                    # select_fn=lambda input: arg_approx(input, threshold),
                    # overlap_fn=calc_trace_side_overlap,
                    overlap_fn=overlap_fn,
                    # overlap_fn=calc_iou,
                    # overlap_fn=calc_class_trace_side_overlap,
                    # overlap_fn=calc_class_trace_side_overlap_norm,
                    # overlap_fn=calc_weighted_iou,
                    path="metrics/"
                    + path_template.format(
                        # path='alexnet_imagenet_class_overlap_ratio_per_node_{0:.1f}_{1}_{2}.foolbox.csv'.format(
                        threshold,
                        attack_name.value,
                        label_name,
                    ),
                    # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.iou.csv'.format(threshold, attack_name, label),
                    # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.class_side.csv'.format(
                    # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.wo_pool.csv'.format(
                    # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.class_side_norm.csv'.format(
                    # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.weighted_iou.csv'.format(
                    #     threshold, attack_name, label),
                    preprocessing=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    channel_axis=3,
                    image_size=224,
                    class_num=1000,
                    norm_fn=imagenet.normalize_alexnet,
                    data_format="channels_last",
                    per_node=per_node,
                    per_channel=per_channel.value,
                    topk_share_range=topk_share_range.value,
                    topk_calc_range=topk_calc_range,
                    use_weight=use_weight,
                    support_diff=support_diff,
                    threshold=threshold,
                    rank=rank.value,
                    use_point=use_point.value,
                    label=label,
                    # trace_label=trace_label.value,
                )
                # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_error(
                # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_rand(
                # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_top5_rand(
                #     class_trace_fn=lambda class_id: alexnet_imagenet_class_trace_compact(class_id, threshold, label=label),
                #     select_fn=lambda input: arg_approx(input, threshold),
                #     overlap_fn=overlap_fn,
                #     path=path_template.format(threshold, attack_name, label),
                # )

                alexnet_overlap_ratio.save()

                # summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_diff.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_diff_same_class_trace.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_diff_all.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_diff_all_compare.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_diff_all_compare.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top2_diff_all_compare.{key}.csv"
                summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_unique_compare.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_diff_all_compare_online.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_diff_all_compare_filter.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_diff_train.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_weighted_iou.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_weighted_iou_class_0.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_class_0.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_full.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_train_in_trace.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_train_not_merged.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_all.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_error.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_rand.{key}.csv"
                # summary_path_template = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_rand.{key}.csv"

                # key = TraceKey.EDGE
                # # summary_file = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}.{key}.csv".format(
                # summary_file = summary_path_template.format(
                #     # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.iou.csv".format(
                #     # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.class_side.csv".format(
                #     # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.wo_pool.csv".format(
                #     # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.class_side_norm.csv".format(
                #     # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.weighted_iou.csv".format(
                #     threshold=threshold, attack=attack_name, label=label_name, key=key)
                # with open(summary_file, "w") as csv_file:
                #     has_header = False
                #     for overlay_threshold in np.linspace(-1, 1, 201):
                #         # summary = get_overlay_summary(alexnet_overlap_ratio.load(), key, overlay_threshold)
                #         # summary = get_overlay_summary_top1(alexnet_overlap_ratio.load(), key, overlay_threshold)
                #         summary = get_overlay_summary_compare(alexnet_overlap_ratio.load(), key, float(overlay_threshold))
                #         # summary = get_overlay_summary_compare_filter(alexnet_overlap_ratio.load(), key, float(overlay_threshold))
                #         # summary = get_overlay_summary_one_side(alexnet_overlap_ratio.load(), key, overlay_threshold)
                #         if not has_header:
                #             headers = ["attack"] + list(summary.keys())
                #             writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n', fieldnames=headers)
                #             writer.writeheader()
                #             has_header = True
                #         writer.writerow({"attack": attack_name, **summary})
                #
                # summary_file = summary_path_template.format(
                #     threshold=threshold, attack=attack_name, label=label_name, key="detail")
                # get_overlay_summary_compare_detail(summary_file, alexnet_overlap_ratio.load(), from_zero=True).save()
