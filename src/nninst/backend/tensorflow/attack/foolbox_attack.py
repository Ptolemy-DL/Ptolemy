import csv
import random
from functools import partial
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
from foolbox.attacks import (
    FGSM,
    Attack,
    DeepFoolAttack,
    IterativeGradientSignAttack,
    SaliencyMapAttack,
)
from foolbox.criteria import TargetClass
from foolbox.distances import MSE
from foolbox.models import TensorFlowModel
from tensorflow.python.training import saver
from tensorflow.python.training.session_manager import SessionManager

from nninst import mode
from nninst.backend.tensorflow.attack.common import get_overlay_summary, overlap_ratio
from nninst.backend.tensorflow.attack.cw_attack import cw_generate_adversarial_example
from nninst.backend.tensorflow.attack.cw_attacks import CarliniL2
from nninst.backend.tensorflow.dataset import mnist
from nninst.backend.tensorflow.trace.lenet_mnist_class_trace_v2 import (
    lenet_mnist_class_trace,
)
from nninst.backend.tensorflow.utils import new_session_config
from nninst.statistics import calc_trace_side_overlap
from nninst.trace import TraceKey
from nninst.utils.fs import IOAction
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init


def foolbox_generate_adversarial_example(
    label: int,
    create_model,
    input_fn: Callable[[], tf.Tensor],
    attack_fn: Callable[..., Attack],
    model_dir=None,
    checkpoint_path=None,
    preprocessing=(0, 1),
    channel_axis=1,
    bounds=(0, 1),
    **kwargs,
) -> Optional[np.ndarray]:
    # Check that model has been trained.
    if not checkpoint_path:
        checkpoint_path = saver.latest_checkpoint(model_dir)
    if not checkpoint_path:
        raise ValueError(
            "Could not find trained model in model_dir: {}.".format(model_dir)
        )

    with tf.Graph().as_default():
        features = input_fn()
        model = create_model()
        image_tensor = tf.placeholder(features.dtype, features.shape)
        logits = model(image_tensor)
        sm = SessionManager()
        with sm.prepare_session(
            master="",
            saver=tf.train.Saver(),
            checkpoint_filename_with_path=checkpoint_path,
            config=new_session_config(),
        ) as sess:
            with sess.as_default():
                image = sess.run(features)[0]
                attack_model = TensorFlowModel(
                    image_tensor,
                    logits,
                    bounds=bounds,
                    channel_axis=channel_axis,
                    preprocessing=preprocessing,
                )
                attack = attack_fn(attack_model)
                # adversarial_example = attack(image, label=label, **kwargs)
                adversarial = attack(image, label=label, unpack=False)
                if adversarial.image is None:
                    return None
                (
                    predictions,
                    is_adversarial,
                    is_best,
                    distance,
                ) = adversarial.predictions(adversarial.image, return_details=True)
                if is_adversarial:
                    return adversarial.image[np.newaxis]
                else:
                    return None


def foolbox_get_l2_distance(
    input_fn: Callable[[], tf.Tensor],
    example_fn: Callable[..., IOAction[np.ndarray]],
    class_id: int,
    image_id: int,
    attack_name: str,
    bounds=(0, 1),
) -> float:
    example_io = example_fn(
        attack_name=attack_name, class_id=class_id, image_id=image_id,
    )
    adversarial_example = example_io.load()
    if adversarial_example is None:
        return None
    with tf.Graph().as_default():
        features = input_fn()
        with tf.Session(config=new_session_config()) as session:
            image = session.run(features)[0]
    distance = MSE(image, adversarial_example, bounds=bounds)
    if distance.value == 0:
        return None
    else:
        return distance.value


def random_targeted(attack_fn, class_start: int, class_end: int):
    return partial(
        attack_fn,
        criterion=TargetClass(target_class=random.randint(class_start, class_end)),
    )


if __name__ == "__main__":
    # mode.debug()
    mode.distributed()
    # mode.local()
    # ray_init("gpu")
    ray_init()
    threshold = 0.5
    # threshold = 1
    # threshold = 0.8
    attacks = {
        "FGSM": [FGSM],
        "BIM": [IterativeGradientSignAttack],
        "JSMA": [SaliencyMapAttack],
        "DeepFool": [DeepFoolAttack],
        # "DeepFool_full": [DeepFoolAttack, dict(subsample=None)],
        "CWL2": [CarliniL2],
    }

    label = "early"
    # label = "best_in_10"
    # label = "worst_in_10"
    # label = "import"
    # label = "norm"
    print(f"attack model with label {label} using Foolbox")
    for attack_name in [
        "DeepFool",
        "FGSM",
        "BIM",
        "JSMA",
        # "DeepFool_full",
        "CWL2",
    ]:
        for threshold in [
            1.0,
            # 0.9,
            # 0.7,
            # 0.5,
            # 0.3,
            # 0.1,
        ]:
            overlap_fn = calc_trace_side_overlap
            # overlap_fn = calc_trace_side_overlap_both_compact
            per_channel = False
            # per_channel = True
            # lenet_overlap_ratio = overlap_ratio(
            # path_template = 'lenet_mnist_class_channel_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.csv'
            # path_template = 'lenet_mnist_class_channel_overlap_ratio_{0:.1f}_{1}_{2}_top5_diff_all.foolbox.csv'
            # path_template = 'lenet_mnist_class_overlap_ratio_{0:.1f}_{1}_{2}_top5_diff_all.foolbox.csv'
            # path_template = 'lenet_mnist_class_overlap_ratio_{0:.1f}_{1}_{2}_top2_diff_all.foolbox.csv'
            path_template = (
                "lenet_mnist_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.csv"
            )
            # path_template = 'lenet_mnist_class_overlap_ratio_{0:.1f}_{1}_{2}_recover.foolbox.csv'
            # path_template = 'lenet_mnist_class_overlap_ratio_{0:.1f}_{1}_{2}_regen.foolbox.csv'
            # lenet_overlap_ratio = lenet_mnist_overlap_ratio_top5_diff(
            lenet_overlap_ratio = overlap_ratio(
                attack_name=attack_name,
                attack_fn=attacks[attack_name][0],
                generate_adversarial_fn=cw_generate_adversarial_example
                if attack_name.startswith("CW")
                else foolbox_generate_adversarial_example,
                # class_trace_fn=lambda class_id: lenet_mnist_class_trace(class_id, threshold, label=label),
                # class_trace_fn=lambda class_id: lenet_mnist_class_channel_trace_compact(
                # class_trace_fn=lambda class_id: lenet_mnist_class_trace_compact(
                class_trace_fn=lambda class_id: lenet_mnist_class_trace(
                    class_id, threshold, label=label
                ),
                # class_trace_fn=lambda class_id: lenet_mnist_class_trace(class_id, threshold),
                select_fn=lambda input: arg_approx(input, threshold),
                overlap_fn=overlap_fn,
                # overlap_fn=calc_iou,
                # overlap_fn=calc_class_trace_side_overlap,
                # overlap_fn=calc_class_trace_side_overlap_norm,
                # overlap_fn=calc_weighted_iou,
                # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.csv'.format(threshold, attack_name, label),
                path=path_template.format(threshold, attack_name, label),
                # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.iou.csv'.format(threshold, attack_name, label),
                # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.class_side.csv'.format(
                # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.wo_pool.csv'.format(
                # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.class_side_norm.csv'.format(
                # path='lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.weighted_iou.csv'.format(
                #     threshold, attack_name, label),
                per_channel=per_channel,
                preprocessing=(0.1307, 0.3081),
                image_size=28,
                class_num=10,
                norm_fn=mnist.normalize,
                data_format="channels_first",
                **(attacks[attack_name][1] if len(attacks[attack_name]) == 2 else {}),
            )

            lenet_overlap_ratio.save()

            print(f"attack: {attack_name}")
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

            # summary_path_template = "lenet_class_channel_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_diff_all_compare.{key}.csv"
            # summary_path_template = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top5_diff_all_compare.{key}.csv"
            # summary_path_template = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_top2_diff_all_compare.{key}.csv"
            summary_path_template = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}.{key}.csv"
            # summary_path_template = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_recover.{key}.csv"
            # summary_path_template = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}_regen.{key}.csv"
            key = TraceKey.EDGE
            # key = TraceKey.WEIGHT
            # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{attack}_{label}.{key}.csv".format(
            summary_file = summary_path_template.format(
                # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.iou.csv".format(
                # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.class_side.csv".format(
                # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.wo_pool.csv".format(
                # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.class_side_norm.csv".format(
                # summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.weighted_iou.csv".format(
                threshold=threshold,
                attack=attack_name,
                label=label,
                key=key,
            )
            with open(summary_file, "w") as csv_file:
                has_header = False
                for overlay_threshold in np.linspace(-1, 1, 201):
                    summary = get_overlay_summary(
                        lenet_overlap_ratio.load(), key, overlay_threshold
                    )
                    # summary = get_overlay_summary_compare(lenet_overlap_ratio.load(), key, float(overlay_threshold))
                    if not has_header:
                        headers = ["attack"] + list(summary.keys())
                        writer = csv.DictWriter(
                            csv_file,
                            delimiter=",",
                            lineterminator="\n",
                            fieldnames=headers,
                        )
                        writer.writeheader()
                        has_header = True
                    writer.writerow({"attack": attack_name, **summary})

            # summary_file = summary_path_template.format(
            #     threshold=threshold, attack=attack_name, label=label, key="detail")
            # get_overlay_summary_compare_detail(summary_file, lenet_overlap_ratio.load()).save()
