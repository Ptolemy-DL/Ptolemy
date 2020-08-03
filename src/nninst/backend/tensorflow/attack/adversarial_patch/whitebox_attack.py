from nninst.backend.tensorflow.attack.adversarial_patch.constant import MODEL_NAMES
from nninst.backend.tensorflow.attack.adversarial_patch.meta_model import MetaModel
from nninst.backend.tensorflow.attack.adversarial_patch.model_container import (
    ModelContainer,
)
from nninst.backend.tensorflow.attack.adversarial_patch.utils.io import (
    load_obj,
    save_obj,
)

model_targets = MODEL_NAMES
# model_targets = ["alexnet"]
STEPS = 500


def save_patches(steps: int = STEPS, learning_rate=5.0, scale=(0.1, 1.0)):
    MM = MetaModel()
    regular_training_model_to_patch = {}
    x = 0
    for m in model_targets:
        print("Training %s" % m)
        M = MM.nc[m]
        M.reset_patch()
        for i in range(steps):
            x += 1
            loss = M.train_step(scale=scale, learning_rate=learning_rate)
            if i % int(steps / 10) == 0:
                print("[%s] loss: %s" % (i, loss))

        regular_training_model_to_patch[m] = M.patch()

    save_obj(regular_training_model_to_patch, "regular_training_model_to_patch")


def update_patch(
    model_name: str, steps: int = STEPS, learning_rate=5.0, scale=(0.1, 1.0)
):
    regular_training_model_to_patch = load_obj("regular_training_model_to_patch")
    M = ModelContainer.create(model_name)
    print("Training %s" % model_name)
    M.reset_patch()
    for i in range(steps):
        loss = M.train_step(scale=scale, learning_rate=learning_rate)
        if i % int(steps / 10) == 0:
            print("[%s] loss: %s" % (i, loss))
    regular_training_model_to_patch[model_name] = M.patch()
    save_obj(regular_training_model_to_patch, "regular_training_model_to_patch")


def load_patches():
    MM = MetaModel()
    regular_training_model_to_patch = load_obj("regular_training_model_to_patch")
    for m in model_targets:
        M = MM.nc[m]
        M.patch(regular_training_model_to_patch[m])
    return MM


if __name__ == "__main__":
    # save_patches(steps=8000, learning_rate=1, scale=None)
    update_patch("alexnet", steps=8000, learning_rate=1, scale=None)
    # update_patch("alexnet")
