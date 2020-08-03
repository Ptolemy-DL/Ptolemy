from .constant import *
from .model_container import ModelContainer


class MetaModel:
    def __init__(self, verbose=True, peace_mask=None, peace_mask_overlay=0.0):
        self.nc: Dict[str, ModelContainer] = {
            m: ModelContainer.create(
                m,
                verbose=verbose,
                peace_mask=peace_mask,
                peace_mask_overlay=peace_mask_overlay,
            )
            for m in MODEL_NAMES
        }
        self._patch = np.zeros(PATCH_SHAPE)
        self.patch_shape = PATCH_SHAPE

    def patch(self, new_patch=None):
        """Retrieve or set the adversarial patch.

        new_patch: The new patch to set, or None to get current patch.

        Returns: Itself if it set a new patch, or the current patch."""
        if new_patch is None:
            return self._patch

        self._patch = new_patch
        return self

    def reset_patch(self):
        """Reset the adversarial patch to all zeros."""
        self.patch(np.zeros(self.patch_shape))

    def train_step(
        self,
        model=None,
        steps=1,
        images=None,
        target_ys=None,
        learning_rate=5.0,
        scale=None,
        **kwargs
    ):
        """Train the model for `steps` steps.

        Args:
          images: A batch of images to train on, it loads one if not present.
          target_ys: Onehot target vector, defaults to TARGET_ONEHOT
          learning_rate: Learning rate for this train step.
          scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.

        Returns: Loss on the target ys."""

        if model is not None:
            to_train = [self.nc[model]]
        else:
            to_train = self.nc.values()

        losses = []
        for mc in to_train:
            mc.patch(self.patch())
            for i in range(steps):
                loss = mc.train_step(
                    images, target_ys, learning_rate, scale=scale, **kwargs
                )
                losses.append(loss)
            self.patch(mc.patch())
        return np.mean(losses)

    def inference_batch(self, model, images=None, target_ys=None, scale=None):
        """Report loss and label probabilities, and patched images for a batch.

        Args:
          images: A batch of images to train on, it loads if not present.
          target_ys: The target_ys for loss calculation, TARGET_ONEHOT if not present.
          scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.
        """

        mc = self.nc[model]
        mc.patch(self.patch())
        return mc.inference_batch(images, target_ys, scale=scale)
