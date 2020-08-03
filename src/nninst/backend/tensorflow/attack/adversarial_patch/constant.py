from typing import Callable, Dict

import numpy as np
from imagenet_stubs.imagenet_2012_labels import name_to_label
from keras import Model, applications

from nninst.utils.fs import abspath

from .alexnet_adapter import alexnet_adapter

TARGET_LABEL = name_to_label(
    "toaster"
)  # Try "banana", "Pembroke, Pembroke Welsh corgi"
IMAGE_SHAPE = (299, 299, 3)
PATCH_SHAPE = IMAGE_SHAPE
# BATCH_SIZE = 16
BATCH_SIZE = 8
# BATCH_SIZE = 1
REPORT_NUM = 3

# Ensemble of models
NAME_TO_MODEL: Dict[str, Callable[..., Model]] = {
    "alexnet": alexnet_adapter,
    "xception": applications.xception.Xception,
    "vgg16": applications.vgg16.VGG16,
    "vgg19": applications.vgg19.VGG19,
    "resnet50": applications.resnet50.ResNet50,
    "inceptionv3": applications.inception_v3.InceptionV3,
}

MODEL_NAMES = ["alexnet", "resnet50", "xception", "inceptionv3", "vgg16", "vgg19"]

# Data augmentation
# Empirically found that training with a very wide scale range works well
# as a default
SCALE_MIN = 0.3
SCALE_MAX = 1.5
ROTATE_MAX = np.pi / 8  # 22.5 degrees in either direction

MAX_ROTATION = 22.5

# Local data dir to write files to
DATA_DIR = abspath("picture/adversarial_patch")
