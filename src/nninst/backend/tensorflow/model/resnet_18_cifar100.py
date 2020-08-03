import tensorflow as tf

from nninst import Graph
from nninst.backend.tensorflow.graph import build_graph
from nninst.backend.tensorflow.model.resnet_cifar import CifarModel
from nninst.op import Conv2dOp, DenseOp
from nninst.utils.fs import IOAction


class ResNet18Cifar100(CifarModel):
    def __init__(self):
        super().__init__(resnet_size=18, num_classes=100, data_format="channels_first")

    @classmethod
    def create_graph(cls, input_name: str = "IteratorGetNext:0") -> Graph:
        with tf.Session().as_default() as sess:
            input = tf.placeholder(tf.float32, shape=(1, 32, 32, 3))
            new_graph = build_graph([input], [cls()(input)])
            new_graph.rename(new_graph.id(input.name), input_name)
            sess.close()
            return new_graph

    @classmethod
    def graph(cls) -> IOAction[Graph]:
        path = "store/graph/resnet18_cifar100.pkl"
        return IOAction(path, init_fn=lambda: ResNet18Cifar100.create_graph())


if __name__ == "__main__":
    # ResNet18Cifar100.graph().save()
    graph = ResNet18Cifar100.graph().load()
    # layers = graph.layers()
    layers = graph.ops_in_layers(Conv2dOp, DenseOp)
    print(layers)
