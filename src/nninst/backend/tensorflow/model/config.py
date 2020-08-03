from nninst.backend.tensorflow.dataset import cifar10, mnist
from nninst.backend.tensorflow.dataset.cifar100_main import normalize_cifar
from nninst.backend.tensorflow.dataset.imagenet import normalize, normalize_alexnet
from nninst.backend.tensorflow.dataset.imagenet_preprocessing import (
    alexnet_preprocess_image,
    preprocess_image,
)
from nninst.backend.tensorflow.model.resnet_18_cifar100_cdrp import ResNet18Cifar100CDRP

from .alexnet import AlexNet
from .alexnet_cdrp import AlexNetCDRP
from .densenet import DenseNet, normalize_cifar_densenet
from .inception_v4 import InceptionV4
from .lenet import LeNet
from .resnet_18_cifar10 import ResNet18Cifar10
from .resnet_18_cifar100 import ResNet18Cifar100
from .resnet_50 import ResNet50
from .resnet_50_cdrp import ResNet50CDRP
from .vgg_16 import VGG16
from .vgg_16_cdrp import VGG16CDRP
from .vgg_16_cifar10 import VGG16Cifar10


class ModelConfig:
    def __init__(
        self,
        name: str,
        model_dir: str,
        network_class,
        class_num,
        image_num_per_class,
        preprocessing_fn=None,
        normalize_fn=None,
        class_from_zero=True,
    ):
        self.name = name
        self.model_dir = model_dir
        self.network_class = network_class
        self.class_num = class_num
        self.image_num_per_class = image_num_per_class
        self.preprocessing_fn = preprocessing_fn
        self.normalize_fn = normalize_fn
        self.class_from_zero = class_from_zero

    def with_model_dir(self, model_dir) -> "ModelConfig":
        return ModelConfig(
            name=self.name,
            model_dir=model_dir,
            network_class=self.network_class,
            class_num=self.class_num,
            image_num_per_class=self.image_num_per_class,
            preprocessing_fn=self.preprocessing_fn,
            normalize_fn=self.normalize_fn,
            class_from_zero=self.class_from_zero,
        )

    def class_list(self):
        if self.class_from_zero:
            return list(range(self.class_num))
        else:
            return list(range(1, self.class_num + 1))


RESNET_50 = ModelConfig(
    name="resnet_50",
    model_dir="tf/resnet-50-v2/model",
    network_class=ResNet50,
    class_num=1000,
    image_num_per_class=1,
    preprocessing_fn=preprocess_image,
    class_from_zero=False,
    normalize_fn=normalize,
)
DENSENET_CIFAR10 = ModelConfig(
    name="densenet_cifar10",
    model_dir="tf/densenet/model",
    network_class=DenseNet,
    class_num=10,
    image_num_per_class=100,
    normalize_fn=normalize_cifar_densenet,
)
RESNET_18_CIFAR10 = ModelConfig(
    name="resnet_18_cifar10",
    model_dir="tf/resnet-18-cifar10/model_train",
    network_class=ResNet18Cifar10,
    class_num=10,
    image_num_per_class=100,
    normalize_fn=normalize_cifar,
)
RESNET_18_CIFAR100 = ModelConfig(
    name="resnet_18_cifar100",
    model_dir="tf/resnet-18-cifar100/model_train",
    network_class=ResNet18Cifar100,
    class_num=100,
    image_num_per_class=10,
    normalize_fn=normalize_cifar,
)
RESNET_18_CIFAR100_CDRP = ModelConfig(
    name="resnet_18_cifar100",
    model_dir="tf/resnet-18-cifar100/model_train",
    network_class=ResNet18Cifar100CDRP,
    class_num=100,
    image_num_per_class=10,
    normalize_fn=normalize_cifar,
)
RESNET_50_CDRP = ModelConfig(
    name="resnet_50",
    model_dir="tf/resnet-50-v2/model",
    network_class=ResNet50CDRP,
    class_num=1000,
    image_num_per_class=1,
    preprocessing_fn=preprocess_image,
    class_from_zero=False,
    normalize_fn=normalize,
)
LENET = ModelConfig(
    name="lenet",
    model_dir="tf/lenet/model",
    network_class=LeNet,
    class_num=10,
    image_num_per_class=100,
    normalize_fn=mnist.normalize,
)
INCEPTION_V4 = ModelConfig(
    name="inception_v4",
    model_dir="tf/inception_v4/model",
    network_class=InceptionV4,
    class_num=1000,
    image_num_per_class=1,
    class_from_zero=False,
)
VGG_16 = ModelConfig(
    name="vgg_16",
    model_dir="tf/vgg_16/model",
    network_class=VGG16,
    class_num=1000,
    image_num_per_class=1,
    preprocessing_fn=preprocess_image,
    normalize_fn=normalize,
)
VGG_16_CDRP = ModelConfig(
    name="vgg_16",
    model_dir="tf/vgg_16/model",
    network_class=VGG16CDRP,
    class_num=1000,
    image_num_per_class=1,
    preprocessing_fn=preprocess_image,
    normalize_fn=normalize,
)
VGG_16_CIFAR10 = ModelConfig(
    name="vgg_16_cifar10",
    model_dir="",
    network_class=VGG16Cifar10,
    class_num=10,
    image_num_per_class=100,
    normalize_fn=cifar10.normalize,
)
ALEXNET = ModelConfig(
    name="alexnet",
    model_dir="tf/alexnet/model",
    network_class=AlexNet,
    class_num=1000,
    image_num_per_class=1,
    preprocessing_fn=alexnet_preprocess_image,
    normalize_fn=normalize_alexnet,
)
ALEXNET_CDRP = ModelConfig(
    name="alexnet",
    model_dir="tf/alexnet/model",
    network_class=AlexNetCDRP,
    class_num=1000,
    image_num_per_class=1,
    preprocessing_fn=alexnet_preprocess_image,
    normalize_fn=normalize_alexnet,
)
