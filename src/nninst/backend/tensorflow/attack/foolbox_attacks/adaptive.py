import random

import numpy as np
import tensorflow as tf
from foolbox import Adversarial
from foolbox.attacks.base import call_decorator
from foolbox.attacks.iterative_projected_gradient import (
    IterativeProjectedGradientBaseAttack,
    L2ClippingMixin,
    L2DistanceCheckMixin,
)
from foolbox.criteria import TargetClass
from tensorflow.python.ops.losses.losses_impl import Reduction

from nninst.backend.tensorflow.dataset import imagenet_raw
from nninst.backend.tensorflow.model.config import ALEXNET
from nninst.backend.tensorflow.utils import new_session_config
from nninst.dataset.envs import IMAGENET_RAW_DIR


class AdaptiveGradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        image = x
        label = class_
        # assert a.in_bounds(image)
        model = a._model
        image, dpdx = model._process_input(image)
        mimic_image, _ = model._process_input(self.mimic_image)

        mimic_tensors = model._session.run(
            [
                tf.get_default_graph().get_tensor_by_name(layer_name)
                for layer_name in self.layer_names[: self.layer_num]
            ],
            feed_dict={
                model._images: mimic_image[np.newaxis],
                model._label: self.mimic_label,
            },
        )

        g, loss, logits = model._session.run(
            [
                self.new_gradient,
                self.new_loss,
                tf.get_default_graph().get_tensor_by_name("dense_2/BiasAdd:0"),
            ],
            feed_dict={
                model._images: image[np.newaxis],
                model._label: label,
                **{
                    name: value
                    for name, value in zip(self.mimic_tensors, mimic_tensors)
                },
            },
        )
        self.loss_value = loss
        # assert np.argmax(logits[0]) == label
        gradient = model._process_gradient(dpdx, g)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.sqrt(np.sum(np.square(gradient)))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class AdaptiveAttack(
    AdaptiveGradientMixin,
    # L2GradientMixin,
    L2ClippingMixin,
    L2DistanceCheckMixin,
    IterativeProjectedGradientBaseAttack,
):
    def __init__(
        self,
        model=None,
        criterion=TargetClass(-1),
        layer_num=3,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.01,
        iterations=100,
        random_start=False,
        return_early=False,
        retry_times=5,
        use_l2_loss=True,
    ):
        super().__init__(model, criterion)
        self.layer_names = [
            "dense_2/BiasAdd:0",
            "dense_1/Relu:0",
            "dropout/Identity_1:0",
            "dropout/Identity:0",
            "conv2d_4/BiasAdd:0",
            "conv2d_3/BiasAdd:0",
            "conv2d_2/BiasAdd:0",
            "conv2d_1/BiasAdd:0",
            "conv2d/BiasAdd:0",
        ]
        self.layer_num = layer_num
        self.binary_search = binary_search
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.iterations = iterations
        self.random_start = random_start
        self.return_early = return_early
        self.retry_times = retry_times
        self.use_l2_loss = use_l2_loss

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True):

        """
        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        layer_num: int
            Number of layers to mimic.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        retry_times : int
            Number of trials to find better adversarial examples.
        use_l2_loss: bool
            If true, uses L2 loss, otherwise uses cosine loss.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert self.epsilon > 0

        if a.image is not None:
            return

        model = a._model
        losses = []
        self.mimic_tensors = []
        for layer_name in self.layer_names[: self.layer_num]:
            mimic_tensor = tf.placeholder(dtype=tf.float32)
            self.mimic_tensors.append(mimic_tensor)
            layer_tensor = tf.get_default_graph().get_tensor_by_name(layer_name)
            if self.use_l2_loss:
                loss = tf.nn.l2_loss(layer_tensor - mimic_tensor)
            else:
                loss = tf.abs(
                    tf.losses.cosine_distance(
                        tf.math.l2_normalize(tf.reshape(layer_tensor, [-1])),
                        tf.math.l2_normalize(tf.reshape(mimic_tensor, [-1])),
                        axis=0,
                        reduction=Reduction.SUM,
                    )
                )
            losses.append(loss)
        loss = tf.reduce_sum(losses)
        gradients = tf.gradients(loss, model._images)
        assert len(gradients) == 1
        if gradients[0] is None:
            gradients[0] = tf.zeros_like(model._images)
        self.new_gradient = tf.squeeze(gradients[0], axis=0)
        self.new_loss = loss

        best_adv = None
        best_loss = -1
        for i in range(self.retry_times):
            with tf.Graph().as_default():
                with tf.Session(config=new_session_config()) as session:
                    while True:
                        class_id = random.randrange(1, 999)
                        if class_id != a.original_class:
                            image_id = random.randrange(500)
                            model_config = ALEXNET.with_model_dir(
                                "tf/alexnet/model_import"
                            )
                            features = (
                                imagenet_raw.train(
                                    IMAGENET_RAW_DIR,
                                    class_id,
                                    image_id,
                                    normed=False,
                                    class_from_zero=model_config.class_from_zero,
                                    preprocessing_fn=model_config.preprocessing_fn,
                                )
                                .make_one_shot_iterator()
                                .get_next()[0]
                            )
                            mimic_image = session.run(features)[0]
                            logits = model._session.run(
                                model._session.graph.get_tensor_by_name(
                                    "dense_2/BiasAdd:0"
                                ),
                                feed_dict={
                                    model._images: model._process_input(mimic_image)[0][
                                        np.newaxis
                                    ]
                                },
                            )
                            if np.argmax(logits[0]) == class_id:
                                self.mimic_image = mimic_image
                                self.mimic_label = class_id
                                new_adv = Adversarial(
                                    a._model,
                                    TargetClass(class_id),
                                    a.original_image,
                                    a.original_class,
                                    a._distance,
                                )
                                break
            self._run(
                new_adv,
                self.binary_search,
                self.epsilon,
                self.stepsize,
                self.iterations,
                self.random_start,
                self.return_early,
            )
            if new_adv.image is not None:
                predictions, is_adversarial = new_adv.predictions(new_adv.image)
                if is_adversarial:
                    assert np.argmax(predictions) == self.mimic_label
                    if best_adv is None or best_loss > self.loss_value:
                        best_adv = new_adv
                        best_loss = self.loss_value
        if best_adv is not None:
            a._criterion._target_class = best_adv._criterion.target_class()
            a.predictions(best_adv.image)
            return
