import tensorflow as tf
from slim.nets.inception_v4 import inception_v4 as slim_inception_v4
from slim.nets.inception_v4 import inception_v4_arg_scope
from tensorflow.contrib import slim

from nninst import Graph
from nninst.backend.tensorflow.graph import build_graph
from nninst.utils.fs import IOAction, abspath


class InceptionV4:
    def __call__(self, inputs, training=False):
        with slim.arg_scope(inception_v4_arg_scope()):
            return slim_inception_v4(inputs, is_training=training)[0]

    @classmethod
    def create_graph(cls, input_name: str = "IteratorGetNext:0") -> Graph:
        with tf.Session().as_default() as sess:
            input = tf.placeholder(tf.float32, shape=(1, 299, 299, 3))
            new_graph = build_graph([input], [cls()(input)])
            new_graph.rename(new_graph.id(input.name), input_name)
            sess.close()
            return new_graph

    @classmethod
    def graph(cls) -> IOAction[Graph]:
        path = "store/graph/inception_v4.pkl"
        return IOAction(path, init_fn=lambda: InceptionV4.create_graph())


if __name__ == "__main__":
    with tf.Session() as sess:
        input = tf.placeholder(tf.float32, shape=(1, 299, 299, 3))
        model = InceptionV4()
        logits = model(input, training=False)
        summary_writer = tf.summary.FileWriter(abspath("tmp/logs"), sess.graph)
        # graph = build_graph([input], [logits])
    #     graph.print()
    # graph = InceptionV4.create_graph()
    # print()
    # InceptionV4.graph().save()
