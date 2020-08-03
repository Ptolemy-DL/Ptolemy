import csv
import os
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.training import saver

from nninst import mode
from nninst.backend.tensorflow.attack.common import get_overlay_summary, overlap_ratio
from nninst.backend.tensorflow.attack.cw_attacks import (
    CarliniL0,
    CarliniL2,
    CarliniLi,
    CWAttack,
)
from nninst.backend.tensorflow.dataset import mnist
from nninst.backend.tensorflow.trace import lenet_mnist_class_trace
from nninst.backend.tensorflow.utils import new_session_config, restore_scope
from nninst.statistics import calc_trace_side_overlap
from nninst.trace import TraceKey
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init


class LogitsFnWrapper:
    def __init__(self, num_channels, image_size, num_labels, logits_fn):
        self.logits_fn = logits_fn
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_labels = num_labels

    def predict(self, x):
        logits = self.logits_fn(x)
        return logits


def cw_generate_adversarial_example(
    label: int,
    create_model,
    input_fn: Callable[[], tf.Tensor],
    attack_fn: Callable[..., CWAttack],
    model_dir=None,
    checkpoint_path=None,
    norm_fn=None,
    channel_axis=1,
    bounds=(0, 1),
    image_size=28,
    class_num=10,
    transpose_input=False,
    # data_format="channels_first",
    targeted_class: int = -1,
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
        logit_tensor = model(norm_fn(features))
        sm = tf.train.SessionManager()
        with sm.prepare_session(
            master="",
            saver=tf.train.Saver(),
            checkpoint_filename_with_path=checkpoint_path,
            config=new_session_config(),
        ) as sess:
            image, logits = sess.run([features, logit_tensor])
            logits = logits[0]
            predict = np.argmax(logits)
            if predict != label:
                return image

    with tf.Graph().as_default():
        features = input_fn()
        model = create_model()

        with tf.Session(config=new_session_config()) as sess:

            def attack_model(x):
                with restore_scope(sess, checkpoint_path):
                    if norm_fn is not None:
                        x = norm_fn(x)
                    if transpose_input:
                        logits = model(tf.transpose(x, [0, 3, 1, 2]))
                    else:
                        logits = model(x)
                    return logits

            attack = attack_fn(
                model=LogitsFnWrapper(
                    num_channels=channel_axis,
                    image_size=image_size,
                    num_labels=class_num,
                    logits_fn=attack_model,
                ),
                sess=sess,
                targeted=(targeted_class != -1),
                boxmin=bounds[0],
                boxmax=bounds[1],
                **kwargs,
            )
            if transpose_input:
                image = sess.run(tf.transpose(features, [0, 2, 3, 1]))
            else:
                image = sess.run(features)
            adversarial_example = attack.attack(
                image,
                np.expand_dims(
                    (np.arange(class_num) == label).astype(np.float32), axis=0
                )
                if targeted_class == -1
                else np.expand_dims(
                    (np.arange(class_num) == targeted_class).astype(np.float32), axis=0
                ),
            )
            if transpose_input:
                return np.transpose(adversarial_example, (0, 3, 1, 2))
            else:
                return adversarial_example


if __name__ == "__main__":
    # mode.debug()
    # mode.local()
    mode.distributed()
    ray_init()
    threshold = 0.5
    attacks = {"CWL2": [CarliniL2], "CWL0": [CarliniL0], "CWLi": [CarliniLi]}

    label = "early"
    # label = "best_in_10"
    # label = "norm"
    print(f"attack model with label {label} using CW")
    for attack_name in [
        "CWL2",
        # "CWLi",
        # "CWL0",
    ]:
        lenet_overlap_ratio = overlap_ratio(
            attack_fn=attacks[attack_name][0],
            generate_adversarial_fn=cw_generate_adversarial_example,
            class_trace_fn=lambda class_id: lenet_mnist_class_trace(
                class_id, threshold, label=label
            ),
            select_fn=lambda input: arg_approx(input, threshold),
            overlap_fn=calc_trace_side_overlap,
            path="lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.cw.csv".format(
                threshold, attack_name, label
            ),
            norm_fn=mnist.normalize,
            **(attacks[attack_name][1] if len(attacks[attack_name]) == 2 else {}),
        )

        lenet_overlap_ratio.save()
        print(f"attack: {attack_name}")
        # print("edge:")
        summary = get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.EDGE)
        print(summary)
        # print("weight:")
        # print(get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.WEIGHT))
        # print("point:")
        # print(get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.POINT))

        summary_file = "lenet_class_overlap_ratio_summary_{threshold:.1f}_{label}.csv".format(
            threshold=threshold, label=label
        )
        file_exists = os.path.exists(summary_file)
        with open(summary_file, "a") as csv_file:
            headers = ["attack"] + list(summary.keys())
            writer = csv.DictWriter(
                csv_file, delimiter=",", lineterminator="\n", fieldnames=headers
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({"attack": attack_name, **summary})
