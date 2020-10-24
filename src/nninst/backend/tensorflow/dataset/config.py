from typing import Callable

import tensorflow as tf

from nninst.dataset.envs import IMAGENET_RAW_DIR

from . import cifar10, cifar10_main, cifar100_main, imagenet, imagenet_raw, mnist


class DataConfig:
    def __init__(self, data_dir: str, dataset_fn: Callable[..., tf.data.Dataset]):
        self.data_dir = data_dir
        self.dataset_fn = dataset_fn

    def copy(self, dataset_fn) -> "DataConfig":
        return DataConfig(data_dir=self.data_dir, dataset_fn=dataset_fn)


IMAGENET_TRAIN = DataConfig(
    data_dir="/home/yxqiu/data/imagenet", dataset_fn=imagenet.train
)
IMAGENET_TEST = DataConfig(
    data_dir="/home/yxqiu/data/imagenet", dataset_fn=imagenet.test
)
IMAGENET_RAW_TRAIN = DataConfig(
    data_dir=IMAGENET_RAW_DIR, dataset_fn=imagenet_raw.train
)
IMAGENET_RAW_TEST = DataConfig(data_dir=IMAGENET_RAW_DIR, dataset_fn=imagenet_raw.test)
MNIST_TRAIN = DataConfig(data_dir="/home/yxqiu/data/mnist/raw", dataset_fn=mnist.train)
MNIST_TEST = DataConfig(data_dir="/home/yxqiu/data/mnist/raw", dataset_fn=mnist.test)
CIFAR10_TRAIN_VGG16 = DataConfig(data_dir="", dataset_fn=cifar10.train)
CIFAR10_TEST_VGG16 = DataConfig(data_dir="", dataset_fn=cifar10.test)
CIFAR10_TRAIN = DataConfig(
    data_dir="cifar10-raw", dataset_fn=cifar10_main.train
)
CIFAR10_TEST = DataConfig(
    data_dir="cifar10-raw", dataset_fn=cifar10_main.test
)
CIFAR100_TRAIN = DataConfig(
    data_dir="cifar100-raw", dataset_fn=cifar100_main.train
)
CIFAR100_TEST = DataConfig(
    data_dir="cifar100-raw", dataset_fn=cifar100_main.test
)
