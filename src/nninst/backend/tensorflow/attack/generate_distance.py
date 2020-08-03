from functools import partial
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from nninst import mode
from nninst.backend.tensorflow.attack.common import alexnet_imagenet_example
from nninst.backend.tensorflow.attack.foolbox_attack import foolbox_get_l2_distance
from nninst.backend.tensorflow.dataset import imagenet_raw
from nninst.backend.tensorflow.model.config import ALEXNET, ModelConfig
from nninst.dataset.envs import IMAGENET_RAW_DIR
from nninst.utils import filter_not_null
from nninst.utils.fs import CsvIOAction, IOAction
from nninst.utils.ray import ray_init, ray_map


def adversarial_distance(
    model_config: ModelConfig,
    attack_name: str,
    distance_fn: Callable[..., IOAction[np.ndarray]],
    input_fn,
    class_ids: Iterable[int],
    image_ids: Iterable[int],
    cache: bool = True,
) -> CsvIOAction:
    def get_single_distance(class_id, image_id):
        distance = distance_fn(
            class_id=class_id,
            image_id=image_id,
            attack_name=attack_name,
            input_fn=partial(input_fn, class_id=class_id, image_id=image_id,),
        )
        if distance is not None:
            return dict(class_id=class_id, image_id=image_id, distance=distance)
        else:
            return None

    def get_adversarial_distance():
        distances = filter_not_null(
            ray_map(
                get_single_distance,
                [
                    (class_id, image_id)
                    for image_id in image_ids
                    for class_id in class_ids
                ],
                out_of_order=True,
            )
        )
        df = pd.DataFrame(distances)
        return df

    path = f"metrics/adversarial_distance_{attack_name}_{model_config.name}.csv"
    return CsvIOAction(path, init_fn=get_adversarial_distance, cache=cache)


if __name__ == "__main__":
    # mode.debug()
    # mode.distributed()
    mode.local()
    ray_init()

    for distance_fn, attack_name in [
        [foolbox_get_l2_distance, "DeepFool"],
        [foolbox_get_l2_distance, "Adaptive_layer1"],
        [foolbox_get_l2_distance, "Adaptive_layer2"],
        [foolbox_get_l2_distance, "Adaptive_layer3"],
        [foolbox_get_l2_distance, "Adaptive_layer9"],
    ]:
        for model_config, input_fn, example_fn, class_ids, image_ids, arch_args in [
            [
                ALEXNET,
                lambda class_id, image_id: imagenet_raw.train(
                    IMAGENET_RAW_DIR,
                    class_id,
                    image_id,
                    normed=False,
                    class_from_zero=ALEXNET.class_from_zero,
                    preprocessing_fn=ALEXNET.preprocessing_fn,
                )
                .make_one_shot_iterator()
                .get_next()[0],
                alexnet_imagenet_example,
                range(1000),
                # range(50),
                range(1),
                {},
            ],
            # [
            #     resnet_18_cifar100_example,
            #     range(100),
            #     range(10),
            #     dict(
            #         bounds=(0, 255),
            #         channel_axis=3,
            #         image_size=32,
            #         class_num=100,
            #         preprocessing=normalize_cifar_with_grad,
            #         norm_fn=tf.image.per_image_standardization,
            #     ),
            # ],
            # [
            #     resnet_18_cifar10_example,
            #     range(10),
            #     range(100),
            #     dict(
            #         bounds=(0, 255),
            #         channel_axis=3,
            #         image_size=32,
            #         class_num=10,
            #         preprocessing=normalize_cifar_with_grad,
            #         norm_fn=tf.image.per_image_standardization,
            #     ),
            # ],
            # [
            #     densenet_cifar10_example,
            #     range(10),
            #     range(100),
            #     dict(
            #         bounds=(0, 255),
            #         channel_axis=3,
            #         image_size=32,
            #         class_num=10,
            #         preprocessing=normalize_cifar_densenet_with_grad,
            #         norm_fn=lambda image: (image - pp_mean)  / 128.0 - 1,
            #     ),
            # ],
            # [
            #     lenet_mnist_example,
            #     range(10),
            #     # range(1000),
            #     range(100),
            #     dict(preprocessing=(0.1307, 0.3081),
            #          image_size=28,
            #          class_num=10,
            #          norm_fn=mnist.normalize,
            #          transpose_input=True,
            #     ),
            # ],
            # [
            #     resnet_50_imagenet_example,
            #     range(1, 1001),
            #     # range(50),
            #     range(1),
            #     dict(
            #         preprocessing=(_CHANNEL_MEANS, 1),
            #         bounds=(0, 255),
            #         channel_axis=3,
            #         image_size=224,
            #         class_num=1001,
            #         norm_fn=imagenet.normalize,
            #         model_name="resnet_50",
            #     ),
            # ],
            # [
            #     vgg_16_imagenet_example,
            #     range(0, 1000),
            #     # range(50),
            #     range(1),
            #     dict(preprocessing=(_CHANNEL_MEANS, 1),
            #          bounds=(0, 255),
            #          channel_axis=3,
            #          image_size=224,
            #          class_num=1000,
            #          norm_fn=imagenet.normalize,
            #          model_name="vgg_16",
            #     ),
            #  ],
        ]:
            label = None
            # label = "without_dropout"
            adversarial_distance(
                model_config=model_config,
                attack_name=attack_name,
                distance_fn=partial(
                    distance_fn,
                    example_fn=partial(example_fn, label=label),
                    **arch_args,
                ),
                input_fn=input_fn,
                class_ids=class_ids,
                image_ids=image_ids,
                cache=True,
            ).save()
