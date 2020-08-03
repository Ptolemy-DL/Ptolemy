import itertools
from functools import partial

from foolbox.attacks import (
    FGSM,
    DeepFoolAttack,
    IterativeGradientSignAttack,
    SaliencyMapAttack,
)

from nninst import mode
from nninst.backend.tensorflow.attack.adversarial_patch_attack import (
    new_patch_generate_adversarial_example,
    patch_generate_adversarial_example,
)
from nninst.backend.tensorflow.attack.common import (
    alexnet_imagenet_example,
    alexnet_imagenet_example_trace,
    example_similarity,
    generate_example_traces,
    intra_class_similarity,
    lenet_mnist_example_trace,
    resnet_18_cifar10_example_trace,
    resnet_18_cifar100_example_trace,
    resnet_50_imagenet_example,
    resnet_50_imagenet_example_trace,
    vgg_16_imagenet_example,
    vgg_16_imagenet_example_trace,
)
from nninst.backend.tensorflow.attack.cw_attack import cw_generate_adversarial_example
from nninst.backend.tensorflow.attack.cw_attacks import CarliniL2
from nninst.backend.tensorflow.attack.foolbox_attack import (
    foolbox_generate_adversarial_example,
    random_targeted,
)
from nninst.backend.tensorflow.attack.foolbox_attacks.fgsm import TargetedIterativeFGSM
from nninst.backend.tensorflow.attack.random_attack import (
    RandomAttack,
    generate_negative_example,
)
from nninst.backend.tensorflow.dataset import imagenet
from nninst.backend.tensorflow.dataset.imagenet_preprocessing import _CHANNEL_MEANS
from nninst.backend.tensorflow.model import AlexNet
from nninst.backend.tensorflow.model.resnet_18_cifar10 import ResNet18Cifar10
from nninst.backend.tensorflow.trace.alexnet_imagenet_class_trace import (
    alexnet_imagenet_class_trace_compact,
)
from nninst.backend.tensorflow.trace.resnet_18_cifar10_class_trace_v2 import (
    resnet_18_cifar10_class_trace_compact,
)
from nninst.trace import Conv2dOp, DenseOp, get_trace
from nninst.utils.ray import ray_init

if __name__ == "__main__":
    # mode.debug()
    mode.distributed()
    # mode.local()
    ray_init()
    # ray_init("gpu")

    for generate_adversarial_fn, attack_name, attack_fn in [
        [None, "original", None],
        # [foolbox_generate_adversarial_example, "DeepFool", DeepFoolAttack],
        # [foolbox_generate_adversarial_example, "FGSM", FGSM],
        # [foolbox_generate_adversarial_example, "FGSM_targeted", random_targeted(FGSM, 1, 999)],
        # [foolbox_generate_adversarial_example, "FGSM_iterative_targeted", random_targeted(TargetedIterativeFGSM, 1, 999)],
        # [foolbox_generate_adversarial_example, "JSMA", SaliencyMapAttack],
        # [foolbox_generate_adversarial_example, "BIM", IterativeGradientSignAttack],
        # [foolbox_generate_adversarial_example, "Random", RandomAttack],
        # [cw_generate_adversarial_example, "CWL2", CarliniL2],
        # [cw_generate_adversarial_example, "CWL2_confidence=3.5", partial(CarliniL2, confidence=3.5)],
        # [cw_generate_adversarial_example, "CWL2_confidence=14", partial(CarliniL2, confidence=14)],
        # [cw_generate_adversarial_example, "CWL2_confidence=28", partial(CarliniL2, confidence=28)],
        # [partial(cw_generate_adversarial_example, targeted_class=500), "CWL2_target=500", CarliniL2],
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
        for (
            example_trace_fn,
            class_trace_fn,
            class_ids,
            image_ids,
            name,
            label,
            model,
        ) in [
            [
                alexnet_imagenet_example_trace,
                alexnet_imagenet_class_trace_compact,
                range(0, 1000, 100),
                # range(1),
                # range(50),
                range(50),
                "alexnet_imagenet",
                "import",
                AlexNet,
            ],
            # [resnet_18_cifar100_example_trace, range(100), range(10)],
            # [
            #     lenet_mnist_example_trace,
            #     range(10),
            #     # range(1000),
            #     range(100),
            # ],
            [
                resnet_18_cifar10_example_trace,
                resnet_18_cifar10_class_trace_compact,
                range(10),
                range(50),
                "resnet_18_cifar10",
                None,
                ResNet18Cifar10,
            ],
            # [
            #     resnet_50_imagenet_example_trace,
            #     range(1, 1001),
            #     # range(50),
            #     range(1),
            # ],
            # [
            #     vgg_16_imagenet_example_trace,
            #     range(0, 1000),
            #     # range(50),
            #     range(1),
            # ],
        ]:
            for (
                threshold,
                per_channel,
                train,
                class_id,
                layer_name,
            ) in itertools.product(
                [
                    # 1.0,
                    # 0.9,
                    # 0.7,
                    0.5,
                    # 0.3,
                    # 0.1,
                ],
                [
                    # True,
                    False
                ],
                [
                    # True,
                    False,
                ],
                class_ids,
                [
                    None,
                    # *model.graph().load().ops_in_layers(Conv2dOp, DenseOp),
                ],
            ):
                example_similarity(
                    example_trace_fn=partial(
                        example_trace_fn,
                        attack_fn=attack_fn,
                        generate_adversarial_fn=generate_adversarial_fn,
                        trace_fn=get_trace,
                        threshold=threshold,
                        per_channel=per_channel,
                        # cache=False,
                        select_seed_fn=None,
                        entry_points=None,
                        train=train,
                    ),
                    class_trace_fn=lambda class_id: class_trace_fn(
                        class_id, threshold, label=label,
                    ),
                    name=name,
                    threshold=threshold,
                    class_id=class_id,
                    image_ids=image_ids,
                    attack_name=attack_name,
                    label=label,
                    layer_name=layer_name,
                    # cache=False,
                ).save()
