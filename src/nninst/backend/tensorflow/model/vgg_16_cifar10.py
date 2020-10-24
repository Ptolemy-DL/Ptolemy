import tensorflow as tf
from tensorflow.python.keras import backend as K

from nninst import Graph
from nninst.backend.tensorflow.graph import build_graph
from nninst.backend.tensorflow.model.cifar10vgg import cifar10vgg
from nninst.utils.fs import IOAction, abspath

K.set_learning_phase(0)


class VGG16Cifar10:
    def __init__(self):
        self.model = None

    def __call__(self, inputs, training=False):
        if self.model is None:
            self.model = cifar10vgg(train=False).model
        return self.model(inputs, training=training).op.inputs[0]

    @classmethod
    def create_graph(cls, input_name: str = "IteratorGetNext:0") -> Graph:
        session = tf.keras.backend.get_session()
        with session.as_default():
            input = tf.placeholder(tf.float32, shape=(1, 32, 32, 3))
            new_graph = build_graph([input], [cls()(input)])
            new_graph.rename(new_graph.id(input.name), input_name)
            return new_graph

    @classmethod
    def graph(cls) -> IOAction[Graph]:
        path = "store/graph/vgg_16_cifar10.pkl"
        return IOAction(path, init_fn=lambda: VGG16Cifar10.create_graph())


if __name__ == "__main__":
    # with tf.Session() as sess:
    #     input = tf.placeholder(tf.float32, shape=(1, 32, 32, 3))
    #     model = VGG16Cifar10()
    #     logits = model(input, training=False)
    #     summary_writer = tf.summary.FileWriter(abspath("tmp/logs"), sess.graph)
    VGG16Cifar10.graph().save()
    # graph = VGG16Cifar10.graph().load()
    # graph.print()
