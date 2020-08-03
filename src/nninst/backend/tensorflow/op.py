import tensorflow as tf

from nninst import Graph, Operation

__all__ = ["TensorFlowOperation"]


class TensorFlowOperation(Operation):
    def __init__(self, graph: Graph, op: tf.Operation):
        super().__init__(graph, op.name)
        self.tf_op = op
