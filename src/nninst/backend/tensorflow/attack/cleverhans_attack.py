from typing import Callable

import numpy as np
import tensorflow as tf
from cleverhans.attacks import (
    Attack,
    BasicIterativeMethod,
    CarliniWagnerL2,
    DeepFool,
    FastGradientMethod,
    SaliencyMapMethod,
)
from cleverhans.model import Model
from tensorflow.python.training import saver

from nninst import mode
from nninst.backend.tensorflow.dataset import mnist
from nninst.backend.tensorflow.trace import lenet_mnist_class_trace
from nninst.backend.tensorflow.utils import new_session_config, restore_scope
from nninst.statistics import calc_trace_side_overlap
from nninst.trace import TraceKey
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init

from .common import get_overlay_summary, overlap_ratio


class AttackModel(Model):
    def __init__(self, image: tf.Tensor, logits: tf.Tensor):
        super().__init__()
        self._image = image
        self._logits = logits

    def fprop(self, x):
        return {"logits": self._logits}


class LogitsFnWrapper(Model):
    def __init__(self, logits_fn):
        super().__init__()
        self.logits_fn = logits_fn

    def get_layer_names(self):
        return ["logits", "probs"]

    def fprop(self, x):
        logits = self.logits_fn(x)
        return {"logits": logits, "probs": tf.nn.softmax(logits)}


def generate_adversarial_example(
    label,
    create_model,
    input_fn: Callable[[], tf.Tensor],
    attack_fn: Callable[..., Attack],
    model_dir=None,
    checkpoint_path=None,
    norm_fn=None,
    **kwargs,
) -> np.ndarray:
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
        # image_tensor = tf.placeholder(features.dtype, features.shape)
        with tf.Session(config=new_session_config()) as sess:

            def attack_model(x):
                with restore_scope(sess, checkpoint_path):
                    if norm_fn is not None:
                        x = norm_fn(x)
                    logits = model(x)
                    return logits

            attack = attack_fn(model=LogitsFnWrapper(attack_model), sess=sess)
            # adversarial_example = attack.generate(image_tensor, **kwargs)
            # adversarial_example = attack.generate(features, **kwargs)

            # with tf.Graph().as_default():
            #     create_model()(input_fn())
            #     variable_names = [variable.name for variable in tf.global_variables()]
            # train_saver = tf.train.Saver(var_list=[variable
            #                                        for variable in tf.global_variables()
            #                                        if variable.name in variable_names])
            # train_saver.restore(sess, checkpoint_path)

            # initialize_uninitialized_vars(sess)
            # return sess.run(adversarial_example, feed_dict={image_tensor: sess.run(features)})
            adversarial_example = sess.run(attack.generate(features, **kwargs))
            return adversarial_example


if __name__ == "__main__":
    mode.debug()
    ray_init()
    threshold = 0.5

    attacks = {
        "FGSM": [FastGradientMethod, dict(clip_min=0.0, clip_max=1.0)],
        "BIM": [BasicIterativeMethod, dict(clip_min=0.0, clip_max=1.0, eps_iter=0.001)],
        "JSMA": [SaliencyMapMethod, dict(clip_min=0.0, clip_max=1.0, theta=0.1)],
        "DeepFool": [
            DeepFool,
            dict(
                clip_min=0.0, clip_max=1.0, max_iter=100, nb_candidate=10, overshoot=0
            ),
        ],
        "CWL2": [CarliniWagnerL2],
    }

    for attack_name in [
        # "FGSM",
        # "BIM",
        "JSMA",
        # "DeepFool",
        # "CWL2",
    ]:
        lenet_overlap_ratio = overlap_ratio(
            attack_fn=attacks[attack_name][0],
            generate_adversarial_fn=generate_adversarial_example,
            class_trace_fn=lambda class_id: lenet_mnist_class_trace(
                class_id, threshold
            ),
            select_fn=lambda input: arg_approx(input, threshold),
            overlap_fn=calc_trace_side_overlap,
            path="lenet_class_overlap_ratio_{:.1f}_{}.cleverhans.csv".format(
                threshold, attack_name
            ),
            norm_fn=mnist.normalize,
            **(attacks[attack_name][1] if len(attacks[attack_name]) == 2 else {}),
        )

        lenet_overlap_ratio.save()
        print(f"attack: {attack_name}")
        print("edge:")
        print(get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.EDGE))
        print("weight:")
        print(get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.WEIGHT))
        print("point:")
        print(get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.POINT))
