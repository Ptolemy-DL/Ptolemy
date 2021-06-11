from functools import partial

import tensorflow as tf
from foolbox.attacks import (
    FGSM,
    DeepFoolAttack,
    IterativeGradientSignAttack,
    SaliencyMapAttack,
)

from nninst import mode
from nninst.backend.tensorflow.attack.common import (
    alexnet_imagenet_example,
    densenet_cifar10_example,
    generate_examples,
    lenet_mnist_example,
    resnet_18_cifar10_example,
    resnet_18_cifar100_example,
    resnet_50_imagenet_example,
    vgg_16_imagenet_example,
)
from nninst.backend.tensorflow.attack.cw_attack import cw_generate_adversarial_example
from nninst.backend.tensorflow.attack.cw_attacks import CarliniL2
from nninst.backend.tensorflow.attack.foolbox_attack import (
    foolbox_generate_adversarial_example,
    random_targeted,
)
from nninst.backend.tensorflow.attack.foolbox_attacks.adaptive import AdaptiveAttack
from nninst.backend.tensorflow.attack.foolbox_attacks.fgsm import (
    TargetedFGSM,
    TargetedIterativeFGSM,
)
from nninst.backend.tensorflow.attack.random_attack import (
    RandomAttack,
    generate_negative_example,
)
from nninst.backend.tensorflow.dataset import imagenet
from nninst.backend.tensorflow.dataset.cifar100_main import (
    normalize_cifar,
    normalize_cifar_with_grad,
)
from nninst.backend.tensorflow.dataset.imagenet_preprocessing import _CHANNEL_MEANS
from nninst.backend.tensorflow.model.densenet import (
    normalize_cifar_densenet,
    normalize_cifar_densenet_with_grad,
    pp_mean,
)
from nninst.utils.ray import ray_init

if __name__ == "__main__":
    # mode.debug()
    # mode.distributed()
    mode.local()
    ray_init()
    # ray_init("gpu")

    for generate_adversarial_fn, attack_name, attack_fn in [
        # [foolbox_generate_adversarial_example, "Adaptive_layer1", partial(AdaptiveAttack, layer_num=1)],
        # [foolbox_generate_adversarial_example, "Adaptive_layer2", partial(AdaptiveAttack, layer_num=2)],
        # [foolbox_generate_adversarial_example, "Adaptive_layer4", partial(AdaptiveAttack, layer_num=4)],
        # [foolbox_generate_adversarial_example, "Adaptive_layer5", partial(AdaptiveAttack, layer_num=5)],
        # [foolbox_generate_adversarial_example, "Adaptive_layer6", partial(AdaptiveAttack, layer_num=6)],
        # [foolbox_generate_adversarial_example, "Adaptive_layer7", partial(AdaptiveAttack, layer_num=7)],
        # [foolbox_generate_adversarial_example, "Adaptive_layer9", partial(AdaptiveAttack, layer_num=9)],
        # [foolbox_generate_adversarial_example, "Adaptive_layer3", partial(AdaptiveAttack, layer_num=3)],
        # [foolbox_generate_adversarial_example, "Adaptive_layer8", partial(AdaptiveAttack, layer_num=8)],
        # [foolbox_generate_adversarial_example, "Adaptive_cos_layer9", partial(AdaptiveAttack, use_l2_loss=False, layer_num=9)],
        # [
        #     foolbox_generate_adversarial_example,
        #     "Adaptive_cos_layer1",
        #     partial(AdaptiveAttack, use_l2_loss=False, layer_num=1),
        # ],
        # [
        #     foolbox_generate_adversarial_example,
        #     "Adaptive_cos_layer3",
        #     partial(AdaptiveAttack, use_l2_loss=False, layer_num=3),
        # ],
        # [
        #     foolbox_generate_adversarial_example,
        #     "Adaptive_cos_layer2",
        #     partial(AdaptiveAttack, use_l2_loss=False, layer_num=2),
        # ],
        # [
        #     foolbox_generate_adversarial_example,
        #     "Adaptive_cos_layer8",
        #     partial(AdaptiveAttack, use_l2_loss=False, layer_num=8),
        # ],
        # [foolbox_generate_adversarial_example, "Adaptive_return_late", partial(AdaptiveAttack, return_early=False)],
        # [foolbox_generate_adversarial_example, "Adaptive_random_start", partial(AdaptiveAttack, random_start=True)],
        # [foolbox_generate_adversarial_example, "Adaptive_iterations_400", partial(AdaptiveAttack, iterations=400, return_early=False)],
        # [foolbox_generate_adversarial_example, "Adaptive_layer4_iterations_400", partial(AdaptiveAttack, iterations=400, return_early=False, layer_num=4)],
        [foolbox_generate_adversarial_example, "FGSM", FGSM],
        [foolbox_generate_adversarial_example, "DeepFool", DeepFoolAttack],
        # [foolbox_generate_adversarial_example, "FGSM_targeted", random_targeted(TargetedFGSM, 1, 999)],
        # [foolbox_generate_adversarial_example, "FGSM_iterative_targeted", random_targeted(TargetedIterativeFGSM, 1, 999)],
        [foolbox_generate_adversarial_example, "JSMA", SaliencyMapAttack],
        [foolbox_generate_adversarial_example, "BIM", IterativeGradientSignAttack],
        # [foolbox_generate_adversarial_example, "Random", RandomAttack],
        [cw_generate_adversarial_example, "CWL2", CarliniL2],
        # [cw_generate_adversarial_example, "CWL2_confidence=3.5", partial(CarliniL2, confidence=3.5)],
        # [cw_generate_adversarial_example, "CWL2_confidence=14", partial(CarliniL2, confidence=14)],
        # [cw_generate_adversarial_example, "CWL2_confidence=28", partial(CarliniL2, confidence=28)],
        # [
        #     partial(cw_generate_adversarial_example, targeted_class=500),
        #     "CWL2_target=500",
        #     CarliniL2,
        # ],
        # [partial(cw_generate_adversarial_example, targeted_class=500), "CWL2_confidence=28_target=500",
        #  partial(CarliniL2, confidence=28)],
        # [cw_generate_adversarial_example, "CWL0", CarliniL0],
        # [cw_generate_adversarial_example, "CWLi", CarliniLi],
        # [patch_generate_adversarial_example, "patch", None],
        # [partial(patch_generate_adversarial_example, scale=0.1), "patch_scale=0.1", None],
        # [partial(patch_generate_adversarial_example, scale=0.2), "patch_scale=0.2", None],
        # [partial(patch_generate_adversarial_example, scale=0.3), "patch_scale=0.3", None],
        # [partial(patch_generate_adversarial_example, scale=0.4), "patch_scale=0.4", None],
        # [partial(patch_generate_adversarial_example, scale=0.5), "patch_scale=0.5", None],
        # [partial(new_patch_generate_adversarial_example, scale=0.1), "new_patch_scale=0.1", None],
        # [partial(new_patch_generate_adversarial_example, scale=0.2), "new_patch_scale=0.2", None],
        # [partial(new_patch_generate_adversarial_example, scale=0.3), "new_patch_scale=0.3", None],
        # [partial(new_patch_generate_adversarial_example, scale=0.4), "new_patch_scale=0.4", None],
        # [partial(new_patch_generate_adversarial_example, scale=0.5), "new_patch_scale=0.5", None],
        # [partial(generate_negative_example, attack_name="negative_example_out_of_top5"),
        #  "negative_example_out_of_top5", None],
    ]:
        for example_fn, class_ids, image_ids, arch_args in [
            # [
            #     alexnet_imagenet_example,
            #     range(1000),
            #     # range(50),
            #     range(1),
            #     dict(
            #         preprocessing=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            #         channel_axis=3,
            #         image_size=224,
            #         class_num=1000,
            #         norm_fn=imagenet.normalize_alexnet,
            #         model_name="alexnet",
            #     ),
            # ],
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
            [
                vgg_16_imagenet_example,
                range(0, 1000),
                # range(50),
                range(1),
                dict(
                    preprocessing=(_CHANNEL_MEANS, 1),
                    bounds=(0, 255),
                    channel_axis=3,
                    image_size=224,
                    class_num=1000,
                    norm_fn=imagenet.normalize,
                    model_name="vgg_16",
                ),
            ],
        ]:
            label = None
            # label = "without_dropout"
            generate_examples(
                example_fn=partial(
                    example_fn,
                    attack_fn=attack_fn,
                    generate_adversarial_fn=partial(
                        generate_adversarial_fn, **arch_args
                    ),
                    label=label,
                ),
                class_ids=class_ids,
                image_ids=image_ids,
                attack_name=attack_name,
                cache=True,
                # cache=False,
            )
