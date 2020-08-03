from typing import Optional

import numpy as np

from nninst.backend.tensorflow.dataset.imagenet import recover_from_alexnet

from .adversarial_patch.loader import image_loader
from .adversarial_patch.model_container import ModelContainer
from .adversarial_patch.utils.io import load_obj


def patch_generate_adversarial_example(
    label: int, model_name: str, scale: int = 0.5, **kwargs
) -> Optional[np.ndarray]:
    regular_training_model_to_patch = load_obj("regular_training_model_to_patch")
    model_container = ModelContainer.create(model_name, batch_size=1)
    model_container.patch(regular_training_model_to_patch[model_name])
    adversarial_example = model_container.adversarial_input(
        [image_loader.get_image(label)], scale=scale
    )
    return adversarial_example


def new_patch_generate_adversarial_example(
    label: int, model_name: str, scale: int = 0.5, **kwargs
) -> Optional[np.ndarray]:
    regular_training_model_to_patch = load_obj("regular_training_model_to_patch")
    model_container = ModelContainer.create(model_name, batch_size=1)
    model_container.patch(regular_training_model_to_patch[model_name])
    adversarial_example = model_container.adversarial_input(
        [image_loader.get_image(label)], scale=scale
    )
    assert model_name == "alexnet"
    if model_name == "alexnet":
        adversarial_example = recover_from_alexnet(adversarial_example)
    return adversarial_example
