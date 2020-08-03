from itertools import product

from nninst import mode
from nninst.backend.tensorflow.attack.common import (
    alexnet_imagenet_adversarial_example_trace,
    alexnet_imagenet_adversarial_example_trace_of_original_class,
    alexnet_imagenet_example_trace_of_target_class,
    alexnet_imagenet_example_trace_old,
    generate_traces,
)
from nninst.utils.ray import ray_init

if __name__ == "__main__":
    # mode.debug()
    mode.distributed()
    # mode.local()
    ray_init()
    # ray_init("gpu")

    for trace_fn, attack_name, threshold in product(
        [
            alexnet_imagenet_example_trace_old,
            alexnet_imagenet_example_trace_of_target_class,
            alexnet_imagenet_adversarial_example_trace,
            alexnet_imagenet_adversarial_example_trace_of_original_class,
        ],
        [
            "DeepFool",
            "FGSM",
            "BIM",
            "JSMA",
            "CWL2",
            "CWL2_confidence=3.5",
            "CWL2_confidence=14",
            "CWL2_confidence=28",
            "CWL2_target=500",
            "CWL2_confidence=28_target=500",
        ],
        [0.5],
    ):
        print(f"start to generate traces for attack {attack_name}")
        for class_ids, image_ids in [
            [
                range(1000),
                # range(50),
                range(1),
            ],
            # [lenet_mnist_example,
            #  range(10),
            #  # range(1000),
            #  range(100),
            #  dict(preprocessing=(0.1307, 0.3081),
            #       image_size=28,
            #       class_num=10,
            #       norm_fn=mnist.normalize,
            #       data_format="channels_first")],
            # [resnet_50_imagenet_example,
            #  range(1, 1001),
            #  # range(50),
            #  range(1),
            #  dict(preprocessing=(_CHANNEL_MEANS, 1),
            #       bounds=(0, 255),
            #       channel_axis=3,
            #       image_size=224,
            #       class_num=1001,
            #       norm_fn=imagenet.normalize,
            #       data_format="channels_last")],
        ]:
            generate_traces(
                trace_fn=trace_fn,
                attack_name=attack_name,
                class_ids=class_ids,
                image_ids=image_ids,
                threshold=threshold,
            )
