from typing import Optional

import numpy as np
import tensorflow as tf
from foolbox.attacks import Attack
from foolbox.attacks.base import call_decorator

from nninst.backend.tensorflow.dataset import imagenet_raw
from nninst.backend.tensorflow.dataset.imagenet_preprocessing import (
    alexnet_preprocess_image,
)
from nninst.backend.tensorflow.model import AlexNet
from nninst.backend.tensorflow.trace import get_rank
from nninst.backend.tensorflow.utils import new_session_config
from nninst.dataset.envs import IMAGENET_RAW_DIR
from nninst.utils.fs import abspath


class RandomAttack(Attack):
    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True):
        image = input_or_adv.original_image
        min_, max_ = input_or_adv.bounds()
        while True:
            perturbed = np.random.uniform(min_, max_, image.shape).astype(np.float32)
            _, is_adversarial = input_or_adv.predictions(perturbed)
            if is_adversarial:
                return


def generate_negative_example(
    label: int, model_name: str, attack_name: str, **kwargs
) -> Optional[np.ndarray]:
    assert model_name == "alexnet"
    data_dir = IMAGENET_RAW_DIR
    model_dir = abspath("tf/alexnet/model_import")
    create_model = lambda: AlexNet()
    adversarial_image_id = 1
    while True:
        adversarial_input_fn = lambda: imagenet_raw.test(
            data_dir,
            label,
            adversarial_image_id,
            class_from_zero=True,
            preprocessing_fn=alexnet_preprocess_image,
        )
        try:
            adversarial_predicted_label_rank = get_rank(
                class_id=label,
                create_model=create_model,
                input_fn=adversarial_input_fn,
                model_dir=model_dir,
            )
        except IndexError:
            return None
        if adversarial_predicted_label_rank == 0:
            adversarial_image_id += 1
        else:
            if attack_name == "negative_example":
                stop = True
            elif attack_name == "negative_example_top5":
                if adversarial_predicted_label_rank < 5:
                    stop = True
                else:
                    stop = False
            elif attack_name == "negative_example_out_of_top5":
                if adversarial_predicted_label_rank >= 5:
                    stop = True
                else:
                    stop = False
            else:
                raise RuntimeError()
            if stop:
                break
            else:
                adversarial_image_id += 1
    with tf.Session(config=new_session_config()) as sess:
        adversarial_example, _ = sess.run(
            imagenet_raw.test(
                data_dir,
                label,
                adversarial_image_id,
                normed=False,
                class_from_zero=True,
                preprocessing_fn=alexnet_preprocess_image,
            )
        )

    return adversarial_example[0]
