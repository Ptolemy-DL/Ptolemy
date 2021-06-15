import os

import numpy as np
from foolbox.attacks import (
    FGSM,
    DeepFoolAttack,
    IterativeGradientSignAttack,
    SaliencyMapAttack,
)

from nninst import mode
from nninst.backend.tensorflow.attack.common import (
    get_overlay_summary_compare,
    get_overlay_summary_compare_detail,
    resnet_50_imagenet_overlap_ratio_top5_diff,
)
from nninst.backend.tensorflow.attack.cw_attack import cw_generate_adversarial_example
from nninst.backend.tensorflow.attack.cw_attacks import CarliniL2
from nninst.backend.tensorflow.attack.foolbox_attack import (
    foolbox_generate_adversarial_example,
)
from nninst.backend.tensorflow.dataset import imagenet
from nninst.backend.tensorflow.dataset.imagenet_preprocessing import _CHANNEL_MEANS
from nninst.backend.tensorflow.trace.resnet_50_imagenet_class_trace_v3 import (
    resnet_50_imagenet_class_trace_compact,
)
from nninst.statistics import calc_trace_side_overlap_both_compact
from nninst.trace import TraceKey
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


if __name__ == "__main__":
    # mode.debug()
    # mode.distributed()
    mode.local()
    # ray_init("gpu")
    # ray_init("dell")
    ray_init()
    threshold = 0.5
    # threshold = 1
    # threshold = 0.8
    attacks = {
        "FGSM": [FGSM],
        "BIM": [IterativeGradientSignAttack],
        "JSMA": [SaliencyMapAttack],
        "DeepFool": [DeepFoolAttack],
        "DeepFool_full": [DeepFoolAttack, dict(subsample=None)],
        "CWL2": [CarliniL2],
    }

    label = None
    # label = "best_in_10"
    # label = "worst_in_10"
    # label = "import"
    # label = "norm"
    print(f"attack model with label {label} using Foolbox")
    for attack_name in [
        "DeepFool",
        # "FGSM",
        # "BIM",
        # "JSMA",
        # "DeepFool_full",
        # "CWL2",
    ]:
        for threshold in [
            # 1.0,
            # 0.9,
            # 0.7,
            0.5,
            # 0.3,
            # 0.1,
        ]:
            print(f"attack: {attack_name}")

            # path_template = "resnet_50_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.csv"
            # path_template = "resnet_50_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_train.foolbox.csv"
            # path_template = "resnet_50_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.weighted_iou.csv"
            # path_template = "resnet_50_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_error.foolbox.csv"
            # path_template = "resnet_50_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_rand.foolbox.csv"
            # path_template = "resnet_50_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top5_rand.foolbox.csv"
            # path_template = "resnet_50_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top5.foolbox.csv"
            # path_template = "resnet_50_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_class_1.foolbox.csv"
            # path_template = "resnet_50_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top5_diff_all.foolbox.csv"
            path_template = "resnet_50_imagenet_class_overlap_ratio_{0:.1f}_{1}_{2}_top2_diff_all.foolbox.csv"
            # overlap_fn = calc_trace_side_overlap_compact
            overlap_fn = calc_trace_side_overlap_both_compact
            # overlap_fn = calc_weighted_iou
            # per_channel = True
            per_channel = False
            # resnet_50_overlap_ratio = resnet_50_imagenet_overlap_ratio_top5(
            # resnet_50_overlap_ratio = resnet_50_imagenet_overlap_ratio(
            resnet_50_overlap_ratio = resnet_50_imagenet_overlap_ratio_top5_diff(
                attack_name=attack_name,
                attack_fn=attacks[attack_name][0],
                generate_adversarial_fn=cw_generate_adversarial_example
                if attack_name.startswith("CW")
                else foolbox_generate_adversarial_example,
                class_trace_fn=lambda class_id: resnet_50_imagenet_class_trace_compact(
                    class_id, threshold, label=label
                ),
                # class_trace_fn=lambda class_id: resnet_50_imagenet_class_trace(class_id, threshold, label=label),
                select_fn=lambda input: arg_approx(input, threshold),
                overlap_fn=overlap_fn,
                # overlap_fn=calc_iou,
                # overlap_fn=calc_class_trace_side_overlap,
                # overlap_fn=calc_class_trace_side_overlap_norm,
                # overlap_fn=calc_weighted_iou,
                path=path_template.format(threshold, attack_name, label),
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
                class_num=1001,
                norm_fn=imagenet.normalize,
                data_format="channels_last",
                per_channel=per_channel,
                **(attacks[attack_name][1] if len(attacks[attack_name]) == 2 else {}),
            )
            # resnet_50_overlap_ratio = resnet_50_imagenet_overlap_ratio_error(
            #     class_trace_fn=lambda class_id: resnet_50_imagenet_class_trace_compact(class_id, threshold, label=label),
            #     select_fn=lambda input: arg_approx(input, threshold),
            #     overlap_fn=overlap_fn,
            #     path=path_template.format(threshold, attack_name, label),
            # )
            # resnet_50_overlap_ratio = resnet_50_imagenet_overlap_ratio_rand(
            # resnet_50_overlap_ratio = resnet_50_imagenet_overlap_ratio_top5_rand(
            #     class_trace_fn=lambda class_id: resnet_50_imagenet_class_trace_compact(class_id, threshold, label=label),
            #     select_fn=lambda input: arg_approx(input, threshold),
            #     overlap_fn=overlap_fn,
            #     path=path_template.format(threshold, attack_name, label),
            # )

            # resnet_50_overlap_ratio.save()

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

            # summary_path_template = "resnet_50_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_diff_all_compare.{key}.csv"
            summary_path_template = "resnet_50_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top2_diff_all_compare.{key}.csv"

            # key = TraceKey.EDGE
            # # summary_file = "alexnet_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}.{key}.csv".format(
            # summary_file = summary_path_template.format(
            #     # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.iou.csv".format(
            #     # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.class_side.csv".format(
            #     # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.wo_pool.csv".format(
            #     # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.class_side_norm.csv".format(
            #     # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.weighted_iou.csv".format(
            #     threshold=threshold, attack=attack_name, label=label, key=key)
            # with open(summary_file, "w") as csv_file:
            #     has_header = False
            #     for overlay_threshold in np.linspace(-1, 1, 201):
            #         # summary = get_overlay_summary(alexnet_overlap_ratio.load(), key, overlay_threshold)
            #         # summary = get_overlay_summary_top1(alexnet_overlap_ratio.load(), key, overlay_threshold)
            #         summary = get_overlay_summary_compare(resnet_50_overlap_ratio.load(), key, float(overlay_threshold))
            #         # summary = get_overlay_summary_compare_filter(alexnet_overlap_ratio.load(), key, float(overlay_threshold))
            #         # summary = get_overlay_summary_one_side(alexnet_overlap_ratio.load(), key, overlay_threshold)
            #         if not has_header:
            #             headers = ["attack"] + list(summary.keys())
            #             writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n', fieldnames=headers)
            #             writer.writeheader()
            #             has_header = True
            #         writer.writerow({"attack": attack_name, **summary})

            summary_file = summary_path_template.format(
                threshold=threshold, attack=attack_name, label=label, key="detail"
            )
            get_overlay_summary_compare_detail(
                summary_file, resnet_50_overlap_ratio.load(), from_zero=False
            ).save()

            # for overlay_threshold in np.arange(0, 1.01, 0.01):
            #     # summary = get_overlay_summary(resnet_50_overlap_ratio.load(), TraceKey.EDGE, overlay_threshold)
            #     for key in [TraceKey.EDGE, TraceKey.WEIGHT]:
            #         summary = get_overlay_summary(resnet_50_overlap_ratio.load(), key, overlay_threshold)
            #         # summary = get_overlay_summary_one_side(resnet_50_overlap_ratio.load(), key, overlay_threshold)
            #         # summary_path_template = "resnet_50_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}.csv"
            #         # summary_path_template = "resnet_50_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_train.{key}.csv"
            #         # summary_path_template = "resnet_50_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_train.{key}.weighted_iou.csv"
            #         # summary_path_template = "resnet_50_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_error.{key}.csv"
            #         # summary_path_template = "resnet_50_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_rand.{key}.csv"
            #         # summary_path_template = "resnet_50_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_rand.{key}.csv"
            #         # summary_path_template = "resnet_50_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5.{key}.csv"
            #         summary_path_template = "resnet_50_imagenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_class_1.{key}.csv"
            #         summary_file = summary_path_template.format(
            #             # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.iou.csv".format(
            #             # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.class_side.csv".format(
            #             # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.wo_pool.csv".format(
            #             # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.class_side_norm.csv".format(
            #             # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.weighted_iou.csv".format(
            #             threshold=threshold, attack=attack_name, label=label, key=key)
            #         file_exists = os.path.exists(summary_file)
            #         with open(summary_file, "a") as csv_file:
            #             headers = ["attack"] + list(summary.keys())
            #             writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n', fieldnames=headers)
            #             if not file_exists:
            #                 writer.writeheader()
            #             writer.writerow({"attack": attack_name, **summary})

            # resnet_50_overlap_ratio_per_node = resnet_50_imagenet_overlap_ratio(
            #     attack_fn=attacks[attack_name][0],
            #     generate_adversarial_fn=generate_adversarial_example,
            #     class_trace_fn=lambda class_id: resnet_50_imagenet_class_trace_compact(class_id, threshold, label=label),
            #     # class_trace_fn=lambda class_id: lenet_mnist_class_trace(class_id, threshold),
            #     select_fn=lambda input: arg_approx(input, threshold),
            #     overlap_fn=calc_trace_side_overlap_compact,
            #     # overlap_fn=calc_iou,
            #     # overlap_fn=calc_class_trace_side_overlap,
            #     # overlap_fn=calc_class_trace_side_overlap_norm,
            #     # overlap_fn=calc_weighted_iou,
            #     path='resnet_50_imagenet_class_overlap_ratio_per_node_{0:.1f}_{1}_{2}.foolbox.csv'.format(
            #         threshold, attack_name, label),
            #     # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.iou.csv'.format(threshold, attack_name, label),
            #     # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.class_side.csv'.format(
            #     # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.wo_pool.csv'.format(
            #     # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.class_side_norm.csv'.format(
            #     # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.weighted_iou.csv'.format(
            #     #     threshold, attack_name, label),
            #     preprocessing=(_CHANNEL_MEANS, 1),
            #     per_node=True,
            #     **(attacks[attack_name][1] if len(attacks[attack_name]) == 2 else {}),
            # )
            # resnet_50_overlap_ratio_per_node.save()
