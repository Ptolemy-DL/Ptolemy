from typing import Tuple

from PIL.Image import Image
from torchvision import datasets
from torchvision.datasets import MNIST

from nninst.dataset import Dataset

__all__ = ["train", "test"]


class MnistDataset(Dataset[int]):
    def __init__(self, mnist: MNIST):
        self.mnist = mnist

    @property
    def size(self) -> int:
        return len(self.mnist)

    def image(self, image_id: int) -> Image:
        return self.mnist[image_id][0]

    def label(self, image_id: int) -> int:
        return self.mnist[image_id][1]

    def image_with_label(self, image_id: int) -> Tuple[Image, int]:
        return self.mnist[image_id]


# _mnist_dir = "/state/ssd0/yxqiu/data/mnist/"
_mnist_dir = "/home/yxqiu/data/mnist/"
_train_dataset = datasets.MNIST(_mnist_dir, train=True, download=True)
_test_dataset = datasets.MNIST(_mnist_dir, train=False, download=True)


def train() -> Dataset:
    return MnistDataset(_train_dataset)


def test() -> Dataset:
    return MnistDataset(_test_dataset)
