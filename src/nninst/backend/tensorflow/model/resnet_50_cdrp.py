import tensorflow as tf

from nninst import Graph
from nninst.backend.tensorflow.graph import build_graph
from nninst.backend.tensorflow.model.resnet_imagenet import ImagenetModel
from nninst.utils.fs import IOAction


class ResNet50CDRP(ImagenetModel):
    def __init__(self, with_gates: bool = True):
        super().__init__(
            resnet_size=50,
            data_format="channels_first",
            gated=True,
            with_gates=with_gates,
        )
