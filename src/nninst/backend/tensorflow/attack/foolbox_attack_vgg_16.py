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
from nninst.backend.tensorflow.attack.common import (
    resnet_50_imagenet_real_metrics_per_layer,
    resnet_50_imagenet_real_metrics_per_layer_targeted,
    vgg_16_imagenet_real_metrics_per_layer,
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
from nninst.backend.tensorflow.dataset.imagenet_preprocessing import _CHANNEL_MEANS
from nninst.backend.tensorflow.model.config import VGG_16
from nninst.backend.tensorflow.model.vgg_16 import VGG16
from nninst.backend.tensorflow.trace.resnet_50_imagenet_class_trace_v3 import (
    resnet_50_imagenet_class_trace_compact,
)
from nninst.backend.tensorflow.trace.vgg_16_imagenet_class_trace import (
    vgg_16_imagenet_class_trace_compact,
)
from nninst.statistics import calc_trace_side_overlap_both_compact
from nninst.trace import get_trace
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


if __name__ == "__main__":
    absolute_threshold, cumulative_threshold, type_ = parse_path_generation_args(VGG_16)
    trace_fn, trace_label, trace_type, trace_parameter = type_
    # mode.debug()
    # mode.distributed()
    mode.local()
    # ray_init("dell")
    ray_init()
    # threshold = 0.5
    # threshold = 0.3
    # threshold = 0.1
    # threshold = 1
    # threshold = 0.8
    attacks = {
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
        "negative_example": [None],
        "negative_example_top5": [None],
        "negative_example_out_of_top5": [None],
        "Random": [RandomAttack],
    }

    label = None
    # label = "best_in_10"
    # label = "worst_in_10"
    # label = "import"
    # label = "norm"
    variant = None
    # variant = "intersect"
    use_weight = False
    # use_weight = True
    print(f"attack model with label {label} using Foolbox")
    for attack_name, generate_adversarial_fn in [
        ["normal", None],
        ["DeepFool", foolbox_generate_adversarial_example],
        ["FGSM", foolbox_generate_adversarial_example],
        # ["FGSM_targeted", foolbox_generate_adversarial_example],
        # ["FGSM_iterative_targeted", foolbox_generate_adversarial_example],
        ["BIM", foolbox_generate_adversarial_example],
        ["JSMA", foolbox_generate_adversarial_example],
        ["CWL2", cw_generate_adversarial_example],
        # ["CWL2_confidence=3.5", cw_generate_adversarial_example],
        # ["CWL2_confidence=14", cw_generate_adversarial_example],
        # ["CWL2_confidence=28", cw_generate_adversarial_example],
        # ["CWL2_target=500", cw_generate_adversarial_example],
        # ["CWL2_confidence=28_target=500", cw_generate_adversarial_example],
        # ["patch", patch_generate_adversarial_example],
        # ["patch_scale=0.1", patch_generate_adversarial_example],
        # ["patch_scale=0.2", patch_generate_adversarial_example],
        # ["patch_scale=0.3", patch_generate_adversarial_example],
        # ["patch_scale=0.4", patch_generate_adversarial_example],
        # ["patch_scale=0.5", patch_generate_adversarial_example],
        # ["negative_example", None],
        # ["negative_example_top5", None],
        # ["negative_example_out_of_top5", None],
        # ["Random", foolbox_generate_adversarial_example],
    ]:
        for threshold, topk_share_range, example_num, rank in itertools.product(
            [
                # 1.0,
                # 0.9,
                # 0.7,
                # 0.5,
                cumulative_threshold,
                # 0.3,
                # 0.1,
            ],
            [
                # 2,
                # 3,
                # 5,
                # 6,
                # 7,
                # 8,
                9,
                # 10,
                # 20,
            ],
            [
                # 100,
                # 400,
                # 700,
                # 1000,
                0
            ],
            [
                # None,
                1,
                2,
            ],
        ):
            topk_calc_range = 2
            print(f"attack: {attack_name}")

            if example_num != 0:
                variant = f"n{example_num}"
            else:
                variant = None

            if variant is not None:
                label_name = f"{label}_{variant}"
            else:
                label_name = label

            if use_weight:
                label_name = f"{label_name}_weight"
            if trace_label is not None:
                label_name = f"{label_name}_{trace_label}"
            if rank is not None:
                label_name = f"{label_name}_rank{rank}"

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
            path_template = "vgg_16_imagenet_real_metrics_per_layer_{0:.1f}_{1}_{2}.csv"
            # path_template = "resnet_50_imagenet_real_metrics_per_layer_targeted0_{0:.1f}_{1}_{2}.csv"
            # path_template = "resnet_50_imagenet_real_metrics_per_layer_targeted500_{0:.1f}_{1}_{2}.csv"
            # path_template = "resnet_50_imagenet_real_metrics_per_layer_targeted800_{0:.1f}_{1}_{2}.csv"
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
            per_channel = False
            # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio(
            # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_top5_diff_uint8(
            # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_top5_diff(
            # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_logit_diff(
            # alexnet_overlap_ratio = alexnet_imagenet_ideal_metrics(
            # alexnet_overlap_ratio = alexnet_imagenet_fc_layer_path_ideal_metrics(
            # alexnet_overlap_ratio = alexnet_imagenet_negative_example_ideal_metrics_per_layer(
            # alexnet_overlap_ratio = alexnet_imagenet_ideal_metrics_per_layer(
            vgg_16_overlap_ratio = vgg_16_imagenet_real_metrics_per_layer(
                # resnet_50_overlap_ratio = resnet_50_imagenet_real_metrics_per_layer_targeted(target_class=0)(
                # resnet_50_overlap_ratio = resnet_50_imagenet_real_metrics_per_layer_targeted(target_class=500)(
                # resnet_50_overlap_ratio = resnet_50_imagenet_real_metrics_per_layer_targeted(target_class=800)(
                # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_top5_unique(
                attack_name=attack_name,
                # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_top5(
                attack_fn=attacks[attack_name][0],
                generate_adversarial_fn=generate_adversarial_fn,
                trace_fn=partial(
                    trace_fn,
                    select_fn=lambda input: arg_approx(input, threshold),
                    select_seed_fn=None,
                    entry_points=None,
                ),
                # class_trace_fn=lambda class_id: alexnet_imagenet_class_trace(class_id, threshold, label=label),
                # class_trace_fn=lambda class_id: alexnet_imagenet_class_trace_compact(class_id, threshold, label=label),
                # class_trace_fn=lambda class_id: alexnet_imagenet_class_channel_trace_compact(
                class_trace_fn=lambda class_id: vgg_16_imagenet_class_trace_compact(
                    # class_trace_fn=lambda class_id: alexnet_imagenet_class_unique_trace_compact(
                    # class_trace_fn=lambda class_id: alexnet_imagenet_class_channel_trace(
                    # class_trace_fn=lambda class_id: alexnet_imagenet_class_trace(
                    class_id,
                    threshold,
                    label=label,
                    variant=variant,
                    trace_type=trace_type,
                    trace_parameter=trace_parameter,
                ),
                # class_trace_fn=lambda class_id: lenet_mnist_class_trace(class_id, threshold),
                select_fn=lambda input: arg_approx(input, threshold),
                # overlap_fn=calc_trace_side_overlap,
                overlap_fn=overlap_fn,
                # overlap_fn=calc_iou,
                # overlap_fn=calc_class_trace_side_overlap,
                # overlap_fn=calc_class_trace_side_overlap_norm,
                # overlap_fn=calc_weighted_iou,
                path=path_template.format(
                    # path='alexnet_imagenet_class_overlap_ratio_per_node_{0:.1f}_{1}_{2}.foolbox.csv'.format(
                    threshold,
                    attack_name,
                    label_name,
                ),
                # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.iou.csv'.format(threshold, attack_name, label),
                # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.class_side.csv'.format(
                # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.wo_pool.csv'.format(
                # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.class_side_norm.csv'.format(
                # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.weighted_iou.csv'.format(
                #     threshold, attack_name, label),
                preprocessing=(_CHANNEL_MEANS, 1),
                bounds=(0, 255),
                channel_axis=3,
                image_size=224,
                class_num=1000,
                norm_fn=imagenet.normalize,
                data_format="channels_last",
                per_node=per_node,
                per_channel=per_channel,
                topk_share_range=topk_share_range,
                topk_calc_range=topk_calc_range,
                use_weight=use_weight,
                threshold=threshold,
                rank=rank,
                label=label,
                **(attacks[attack_name][1] if len(attacks[attack_name]) == 2 else {}),
            )
            # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_error(
            # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_rand(
            # alexnet_overlap_ratio = alexnet_imagenet_overlap_ratio_top5_rand(
            #     class_trace_fn=lambda class_id: alexnet_imagenet_class_trace_compact(class_id, threshold, label=label),
            #     select_fn=lambda input: arg_approx(input, threshold),
            #     overlap_fn=overlap_fn,
            #     path=path_template.format(threshold, attack_name, label),
            # )

            vgg_16_overlap_ratio.save()

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
