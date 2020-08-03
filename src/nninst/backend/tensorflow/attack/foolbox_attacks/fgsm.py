from collections import Iterable

import numpy as np
from foolbox.attacks.base import Attack, call_decorator
from foolbox.attacks.gradient import SingleStepGradientBaseAttack
from foolbox.attacks.iterative_gradient import IterativeGradientBaseAttack


class TargetedFGSM(SingleStepGradientBaseAttack):
    """Adds the sign of the gradient to the image, gradually increasing
    the magnitude until the image is misclassified.

    Does not do anything if the model does not have a gradient.

    """

    @call_decorator
    def __call__(
        self, input_or_adv, label=None, unpack=True, epsilons=1000, max_epsilon=1
    ):
        """Adds the sign of the gradient to the image, gradually increasing
        the magnitude until the image is misclassified.

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
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the direction of the sign of
            the gradient or number of step sizes between 0 and max_epsilon
            that should be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        return self._run(a, epsilons=epsilons, max_epsilon=max_epsilon)

    def _gradient(self, a):
        min_, max_ = a.bounds()
        gradient = a.gradient(label=self._default_criterion.target_class())
        gradient = -np.sign(gradient) * (max_ - min_)
        return gradient


class TargetedIterativeFGSM(Attack):
    """Like GradientSignAttack but with several steps for each epsilon.

    """

    @call_decorator
    def __call__(
        self,
        input_or_adv,
        label=None,
        unpack=True,
        epsilons=1,
        max_epsilon=0.1,
        steps=10,
    ):

        """Like GradientSignAttack but with several steps for each epsilon.

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
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the direction of the sign of
            the gradient or number of step sizes between 0 and max_epsilon
            that should be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.
        steps : int
            Number of iterations to run.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        self._run(a, epsilons=epsilons, max_epsilon=max_epsilon, steps=steps)

    def _run(self, a, epsilons, max_epsilon, steps):
        if not a.has_gradient():
            return

        image = a.original_image
        min_, max_ = a.bounds()

        if not isinstance(epsilons, Iterable):
            assert isinstance(epsilons, int)
            max_epsilon_iter = max_epsilon / steps
            epsilons = np.linspace(0, max_epsilon_iter, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            perturbed = image

            for _ in range(steps):
                gradient = self._gradient(a, perturbed)

                perturbed = perturbed + gradient * epsilon
                perturbed = np.clip(perturbed, min_, max_)

                a.predictions(perturbed)
                # we don't return early if an adversarial was found
                # because there might be a different epsilon
                # and/or step that results in a better adversarial

    def _gradient(self, a, x):
        min_, max_ = a.bounds()
        gradient = a.gradient(x, label=a.target_class())
        gradient = -np.sign(gradient) * (max_ - min_)
        return gradient
