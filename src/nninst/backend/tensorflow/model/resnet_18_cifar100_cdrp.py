from nninst.backend.tensorflow.model.resnet_cifar import CifarModel


class ResNet18Cifar100CDRP(CifarModel):
    def __init__(self, with_gates: bool = True):
        super().__init__(
            resnet_size=18,
            num_classes=100,
            data_format="channels_first",
            gated=True,
            with_gates=with_gates,
        )
