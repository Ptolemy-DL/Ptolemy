import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16 as slim_vgg_16
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_arg_scope

from nninst import Graph
from nninst.backend.tensorflow.graph import build_graph
from nninst.utils.fs import IOAction


class VGG16:
    def __call__(self, inputs, training=False):
        with slim.arg_scope(vgg_arg_scope()):
            return slim_vgg_16(inputs, is_training=training)[0]

    @classmethod
    def create_graph(cls, input_name: str = "IteratorGetNext:0") -> Graph:
        with tf.Session().as_default() as sess:
            input = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
            new_graph = build_graph([input], [cls()(input)])
            new_graph.rename(new_graph.id(input.name), input_name)
            sess.close()
            return new_graph

    @classmethod
    def graph(cls) -> IOAction[Graph]:
        path = "store/graph/vgg_16.pkl"
        return IOAction(path, init_fn=lambda: VGG16.create_graph())


if __name__ == "__main__":
    # with tf.Session() as sess:
    #     input = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    #     model = VGG16()
    #     logits = model(input, training=False)
    #     summary_writer = SummaryWriter(abspath("tmp/logs"), sess.graph)
    #     graph = build_graph([input], [logits])
    #     graph.print()
    # graph = InceptionV4.create_graph()
    # print()
    VGG16.graph().save()
