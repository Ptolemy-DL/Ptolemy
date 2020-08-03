import os.path as osp
import time

import keras
import PIL.Image
import tensorflow as tf
from keras import backend as K

from .constant import *
from .loader import image_loader
from .transformation import _transform_vector


def get_peace_mask(shape):
    path = osp.join(DATA_DIR, "peace_sign.png")
    pic = PIL.Image.open(path)
    pic = pic.resize(shape[:2], PIL.Image.ANTIALIAS)
    if path.endswith(".png"):
        ch = 4
    else:
        ch = 3
    pic = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], ch)
    pic = pic / 127.5 - 1
    pic = pic[:, :, 3]

    peace_mask = (pic + 1.0) / 2
    peace_mask = np.expand_dims(peace_mask, 2)
    peace_mask = np.broadcast_to(peace_mask, shape)
    return peace_mask


def _circle_mask(shape, sharpness=40):
    """Return a circular mask of a given shape"""
    assert shape[0] == shape[1], "circle_mask received a bad shape: " + shape

    diameter = shape[0]
    x = np.linspace(-1, 1, diameter)
    y = np.linspace(-1, 1, diameter)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = (xx ** 2 + yy ** 2) ** sharpness

    mask = 1 - np.clip(z, -1, 1)
    mask = np.expand_dims(mask, axis=2)
    mask = np.broadcast_to(mask, shape).astype(np.float32)
    return mask


def _gen_target_ys():
    label = TARGET_LABEL
    y_one_hot = np.zeros(1000)
    y_one_hot[label] = 1.0
    y_one_hot = np.tile(y_one_hot, (BATCH_SIZE, 1))
    return y_one_hot


TARGET_ONEHOT = _gen_target_ys()


class ModelContainer:
    """Encapsulates an Imagenet model, and methods for interacting with it."""

    def __init__(
        self,
        model_name,
        verbose=True,
        peace_mask=None,
        peace_mask_overlay=0.0,
        batch_size: int = BATCH_SIZE,
    ):
        # Peace Mask: None, "Forward", "Backward"
        self.model_name = model_name
        self.graph = tf.Graph()
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=session_config)
        self.peace_mask = peace_mask
        self.patch_shape = self.image_shape
        self._peace_mask_overlay = peace_mask_overlay
        self._batch_size = batch_size
        self.load_model(verbose=verbose)

    @staticmethod
    def create(model_name: str, *args, **kwargs):
        if model_name in ["xception", "inceptionv3", "mobilenet"]:
            return ModelContainer(model_name, *args, **kwargs)
        elif model_name in ["alexnet"]:
            return AlexNetContainer(model_name, *args, **kwargs)
        else:
            return KerasContainer(model_name, *args, **kwargs)

    @property
    def image_shape(self):
        return IMAGE_SHAPE

    def patch(self, new_patch=None):
        """Retrieve or set the adversarial patch.

        new_patch: The new patch to set, or None to get current patch.

        Returns: Itself if it set a new patch, or the current patch."""
        if new_patch is None:
            return self._run(self._clipped_patch)

        self._run(self._assign_patch, {self._patch_placeholder: new_patch})
        return self

    def reset_patch(self):
        """Reset the adversarial patch to all zeros."""
        self.patch(np.zeros(self.patch_shape))

    def train_step(
        self,
        images=None,
        target_ys=None,
        learning_rate=5.0,
        scale=(0.1, 1.0),
        dropout=None,
        patch_disguise=None,
        disguise_alpha=None,
    ):
        """Train the model for one step.

        Args:
          images: A batch of images to train on, it loads one if not present.
          target_ys: Onehot target vector, defaults to TARGET_ONEHOT
          learning_rate: Learning rate for this train step.
          scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.

        Returns: Loss on the target ys."""
        if images is None:
            images = image_loader.get_images()
        if target_ys is None:
            target_ys = TARGET_ONEHOT

        feed_dict = {
            self._image_input: images,
            self._target_ys: target_ys,
            self._learning_rate: learning_rate,
        }

        if patch_disguise is not None:
            if disguise_alpha is None:
                raise ValueError("You need disguise_alpha")
            feed_dict[self.patch_disguise] = patch_disguise
            feed_dict[self.disguise_alpha] = disguise_alpha

        loss, _ = self._run(
            [self._loss, self._train_op], feed_dict, scale=scale, dropout=dropout
        )
        return loss

    def inference_batch(self, images=None, target_ys=None, scale=None):
        """Report loss and label probabilities, and patched images for a batch.

        Args:
          images: A batch of images to train on, it loads if not present.
          target_ys: The target_ys for loss calculation, TARGET_ONEHOT if not present."""
        if images is None:
            images = image_loader.get_images()
        if target_ys is None:
            target_ys = TARGET_ONEHOT

        feed_dict = {self._image_input: images, self._target_ys: target_ys}

        loss_per_example, ps, ims = self._run(
            [self._loss_per_example, self._probabilities, self._patched_input],
            feed_dict,
            scale=scale,
        )
        return loss_per_example, ps, ims

    def adversarial_input(self, images=None, scale=None):
        if images is None:
            images = image_loader.get_images()

        feed_dict = {self._image_input: images}
        return self._run(self._adversarial_input, feed_dict, scale=scale)

    def load_model(self, verbose=True):
        model = NAME_TO_MODEL[self.model_name]
        patch = None
        self._make_model_and_ops(model, patch, verbose)

    def _run(self, target, feed_dict=None, scale=None, dropout=None):
        K.set_session(self.sess)
        if feed_dict is None:
            feed_dict = {}
        feed_dict[self.learning_phase] = False

        if scale is not None:
            if isinstance(scale, (tuple, list)):
                scale_min, scale_max = scale
            else:
                scale_min, scale_max = (scale, scale)
            feed_dict[self.scale_min] = scale_min
            feed_dict[self.scale_max] = scale_max

        if dropout is not None:
            feed_dict[self.dropout] = dropout
        return self.sess.run(target, feed_dict=feed_dict)

    def _make_model_and_ops(self, M: Callable[..., Model], patch_val, verbose):
        start = time.time()
        K.set_session(self.sess)
        with self.sess.graph.as_default():
            self.learning_phase = K.learning_phase()

            self._image_input = keras.layers.Input(shape=self.image_shape)

            self.scale_min = tf.placeholder_with_default(SCALE_MIN, [])
            self.scale_max = tf.placeholder_with_default(SCALE_MAX, [])
            self._scales = tf.random_uniform(
                [self._batch_size], minval=self.scale_min, maxval=self.scale_max
            )

            image_input = self._image_input
            self.patch_disguise = tf.placeholder_with_default(
                tf.zeros(self.patch_shape), shape=self.patch_shape
            )
            self.disguise_alpha = tf.placeholder_with_default(0.0, [])
            patch = tf.get_variable(
                "patch",
                self.patch_shape,
                dtype=tf.float32,
                initializer=tf.zeros_initializer,
            )
            self._patch_placeholder = tf.placeholder(
                dtype=tf.float32, shape=self.patch_shape
            )
            self._assign_patch = tf.assign(patch, self._patch_placeholder)

            modified_patch = patch

            def clip_to_valid_image(x):
                return tf.clip_by_value(x, clip_value_min=-1.0, clip_value_max=1.0)
                # return tf.clip_by_value(x, clip_value_min=0, clip_value_max=255)

            if self.peace_mask == "forward":
                mask = get_peace_mask(self.patch_shape)
                modified_patch = (
                    patch * (1 - mask)
                    - np.ones(self.patch_shape) * mask
                    + (1 + patch) * mask * self._peace_mask_overlay
                )

            self._clipped_patch = clip_to_valid_image(modified_patch)

            self.dropout = tf.placeholder_with_default(1.0, [])
            patch_with_dropout = tf.nn.dropout(modified_patch, keep_prob=self.dropout)
            patched_input = clip_to_valid_image(
                self._random_overlay(image_input, patch_with_dropout, self.image_shape)
            )

            # Since this is a return point, we do it before the Keras color shifts
            # (but after the resize, so we can see what is really going on)
            self._patched_input = patched_input

            patched_input = self.preprocess(patched_input)
            self._adversarial_input = patched_input

            # Labels for our attack (e.g. always a toaster)
            self._target_ys = tf.placeholder(tf.float32, shape=(None, 1000))

            model = self.create_model(M, input_tensor=patched_input, weights="imagenet")
            logits = self.logits(model)

            self._loss_per_example = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self._target_ys, logits=logits
            )
            self._target_loss = tf.reduce_mean(self._loss_per_example)

            self._patch_loss = (
                tf.nn.l2_loss(patch - self.patch_disguise) * self.disguise_alpha
            )

            self._loss = self._target_loss + self._patch_loss

            # Train our attack by only training on the patch variable
            self._learning_rate = tf.placeholder(tf.float32)
            self._train_op = tf.train.GradientDescentOptimizer(
                self._learning_rate
            ).minimize(self._loss, var_list=[patch])

            self._probabilities = self.probabilities(model)

            if patch_val is not None:
                self.patch(patch_val)
            else:
                self.reset_patch()

            elapsed = time.time() - start
            if verbose:
                print(
                    "Finished loading {}, took {:.0f}s".format(self.model_name, elapsed)
                )

    def preprocess(self, inputs):
        # return tf.image.resize_images(inputs, (299, 299)) / 127.5 - 1
        return inputs

    def create_model(self, model_fn, input_tensor: tf.Tensor, weights: str):
        return model_fn(input_tensor=input_tensor, weights=weights)

    def logits(self, model):
        return model.outputs[0].op.inputs[0]

    def probabilities(self, model):
        return model.outputs[0]

    def _pad_and_tile_patch(self, patch, image_shape):
        # Calculate the exact padding
        # Image shape req'd because it is sometimes 299 sometimes 224

        # padding is the amount of space available on either side of the centered patch
        # WARNING: This has been integer-rounded and could be off by one.
        #          See _pad_and_tile_patch for usage
        return tf.stack([patch] * self._batch_size)

    def _random_overlay(self, imgs, patch, image_shape):
        """Augment images with random rotation, transformation.

        Image: BATCHx299x299x3
        Patch: 50x50x3

        """
        # Add padding

        image_mask = _circle_mask(image_shape)

        if self.peace_mask == "backward":
            peace_mask = get_peace_mask(image_shape)
            image_mask = (image_mask * peace_mask).astype(np.float32)
        image_mask = tf.stack([image_mask] * self._batch_size)
        padded_patch = tf.stack([patch] * self._batch_size)

        transform_vecs = []

        def _random_transformation(scale_min, scale_max, width):
            im_scale = np.random.uniform(low=scale_min, high=scale_max)

            padding_after_scaling = (1 - im_scale) * width
            x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
            y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)

            rot = np.random.uniform(-MAX_ROTATION, MAX_ROTATION)

            return _transform_vector(
                width,
                x_shift=x_delta,
                y_shift=y_delta,
                im_scale=im_scale,
                rot_in_degrees=rot,
            )

        for i in range(self._batch_size):
            # Shift and scale the patch for each image in the batch
            random_xform_vector = tf.py_func(
                _random_transformation,
                [self.scale_min, self.scale_max, image_shape[0]],
                tf.float32,
            )
            random_xform_vector.set_shape([8])

            transform_vecs.append(random_xform_vector)

        image_mask = tf.contrib.image.transform(image_mask, transform_vecs, "BILINEAR")
        padded_patch = tf.contrib.image.transform(
            padded_patch, transform_vecs, "BILINEAR"
        )

        inverted_mask = 1 - image_mask
        return imgs * inverted_mask + padded_patch * image_mask


class KerasContainer(ModelContainer):
    # @property
    # def image_shape(self):
    #     return (224, 224, 3)

    def preprocess(self, inputs):
        x = tf.image.resize_images(inputs, (224, 224))
        x = (x + 1) * 127.5
        R, G, B = tf.split(x, 3, 3)
        R -= 123.68
        G -= 116.779
        B -= 103.939
        x = tf.concat([B, G, R], 3)
        return x


class AlexNetContainer(ModelContainer):
    # @property
    # def image_shape(self):
    #     return (224, 224, 3)

    def create_model(self, model_fn, input_tensor: tf.Tensor, weights: str):
        return model_fn(sess=self.sess, input_tensor=input_tensor, weights=weights)

    def logits(self, model):
        return model

    def probabilities(self, model):
        return tf.nn.softmax(model)

    def preprocess(self, inputs):
        x = tf.image.resize_images(inputs, (224, 224))
        x = (x + 1) / 2.0
        # x = x / 255.0
        mean = tf.expand_dims(tf.expand_dims([0.485, 0.456, 0.406], 0), 0)
        std = tf.expand_dims(tf.expand_dims([0.229, 0.224, 0.225], 0), 0)
        return (x - mean) / std
