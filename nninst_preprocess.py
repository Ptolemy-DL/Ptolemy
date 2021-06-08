import argparse
import os
import pdb
from nninst import mode
from nninst.backend.tensorflow.attack.calc_per_layer_metrics import calc_per_layer_metrics
from nninst.backend.tensorflow.attack.generate_example_traces import generate_original_example_traces

from nninst.backend.tensorflow.model.alexnet import AlexNet
from nninst.backend.tensorflow.model.config import ALEXNET, RESNET_18_CIFAR10, RESNET_18_CIFAR100, VGG_16
from nninst.backend.tensorflow.model.resnet_18_cifar10 import ResNet18Cifar10
from nninst.backend.tensorflow.model.resnet_18_cifar100 import ResNet18Cifar100
from nninst.backend.tensorflow.model.vgg_16 import VGG16
from nninst.utils.ray import ray_init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        default="Alexnet",
        help="different networks, pick between Alexnet, Resnet-18 and Vgg16",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Imagenet",
        help="different datasets, pick between Imagenet, Cifar-10 and Cifar-100"
    )
    parser.add_argument(
        "--ray-mode",
        type=str,
        default="local",
        help="different ray mode, pick between local, debug and distributed"
    )
    params, unparsed = parser.parse_known_args()

    if params.ray_mode == "local":
        mode.local()
    elif params.ray_mode == "debug":
        mode.debug()
    elif params.ray_mode == "distributed":
        mode.distributed()
    else:
        raise ValueError(f"{params.ray_mode} is an invalid ray mode")
    ray_init()

    if params.network == "Alexnet":
        if params.dataset == "Imagenet":
            AlexNet.graph().save()
            generate_original_example_traces("alexnet")
            calc_per_layer_metrics(ALEXNET)
        else:
            raise ValueError(f"Network Dataset combination {params.network} + {params.dataset} is not supported yet")
    elif params.network == "Resnet-18":
        if params.dataset == "Cifar-10":
            ResNet18Cifar10.graph().save()
            generate_original_example_traces("resnet18_cifar10")
            calc_per_layer_metrics(RESNET_18_CIFAR10)
        elif params.dataset == "Cifar-100":
            ResNet18Cifar100.graph().save()
            generate_original_example_traces("resnet18_cifar100")
            calc_per_layer_metrics(RESNET_18_CIFAR100)
        else:
            raise ValueError(f"Network Dataset combination {params.network} + {params.dataset} is not supported yet")
    elif params.network == "Vgg16":
        if params.dataset == "Imagenet":
            VGG16.graph().save()
            generate_original_example_traces("vgg16")
            calc_per_layer_metrics(VGG_16)
        else:
            raise ValueError(f"Network Dataset combination {params.network} + {params.dataset} is not supported yet")
    else:
        raise ValueError(f"Network Dataset combination {params.network} + {params.dataset} is not supported yet")


if __name__ == "__main__":
    main()