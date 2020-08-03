import copy
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunArgs, SessionRunHook

from nninst import AttrMap, Graph, Tensor
from nninst.op import Conv2dOp, DenseOp, get_weight
from nninst.trace import TraceKey

from .op import TensorFlowOperation
from .rule import (
    FuseBiasAdd,
    FuseConcatOp,
    FuseConv2d,
    FuseDense,
    FuseMean,
    FusePad,
    FuseReshape,
    FuseTranspose,
    RemoveIdentity,
    RemoveMulForBatchNorm,
    RewriteToAdd,
    RewriteToBatchNorm,
    RewriteToPool,
    RewriteToRelu,
    RewriteToSqueezeOp,
    RewriteToVariable,
)

__all__ = [
    "build_graph",
    "FetchInnerTensorsHook",
    "model_fn_with_fetch_hook",
    "assign_variables",
    "get_variables_from_tf",
    "load_variables_into_tf",
]


def assign_variables(graph: Graph, variable_name_to_value: Dict[str, Any]):
    variables = graph.variables
    for name, variable in variables.items():
        variable.value = variable_name_to_value[name]
    for op in graph.ops:
        if isinstance(op, Conv2dOp):
            kernel = op.kernel.value
            op.kernel.value = np.transpose(kernel, (3, 2, 0, 1))
        if isinstance(op, DenseOp):
            weight = op.weight.value
            op.weight.value = np.transpose(weight, (1, 0))


def get_variables_from_tf(
    graph: Graph, tf_graph: tf.Graph, session: tf.Session
) -> Dict[str, np.ndarray]:
    return session.run(
        {name: tf_graph.get_tensor_by_name(name) for name, _ in graph.variables.items()}
    )


def load_variables_into_tf(
    tf_graph: tf.Graph,
    variable_name_to_value: Dict[str, np.ndarray],
    session: tf.Session,
):
    with tf_graph.as_default():
        for variable in tf.global_variables():
            if variable.name in variable_name_to_value:
                variable.load(variable_name_to_value[variable.name], session=session)


class MaskWeightHook(SessionRunHook):
    def __init__(self, graph: Graph):
        self.graph = graph

    def after_create_session(self, session, coord):
        tf_graph = tf.get_default_graph()
        graph = self.graph
        variables = get_variables_from_tf(graph, tf_graph, session)
        for op in graph.ops:
            if TraceKey.WEIGHT in op.attrs:
                weight_name = get_weight(op).name
                weight = variables[weight_name]
                weight_shape_in_trace = op.attrs[TraceKey.WEIGHT_SHAPE]
                traced_weight = np.unravel_index(
                    TraceKey.to_array(op.attrs[TraceKey.WEIGHT]), weight_shape_in_trace
                )
                if isinstance(op, Conv2dOp):
                    traced_weight = tuple(
                        [traced_weight[axis] for axis in [2, 3, 1, 0]]
                    )
                elif isinstance(op, DenseOp):
                    traced_weight = tuple([traced_weight[axis] for axis in [1, 0]])
                else:
                    raise RuntimeError()
                mask = np.ones(weight.shape, dtype=np.int32)
                mask[traced_weight] = 0
                weight[mask.astype(np.bool)] = 0
        load_variables_into_tf(tf_graph, variables, session)


class MaskWeightWithTraceHook(SessionRunHook):
    def __init__(self, graph: Graph, trace: AttrMap):
        self.graph = graph
        self.trace = trace

    def after_create_session(self, session, coord):
        tf_graph = tf.get_default_graph()
        graph = self.graph
        variables = get_variables_from_tf(graph, tf_graph, session)
        for op in graph.ops:
            trace_op = self.trace.ops[op.name]
            if TraceKey.WEIGHT in trace_op:
                weight_name = get_weight(op).name
                weight = variables[weight_name]
                weight_shape_in_trace = trace_op[TraceKey.WEIGHT_SHAPE]
                traced_weight = np.unravel_index(
                    np.nonzero(np.unpackbits(trace_op[TraceKey.WEIGHT])),
                    weight_shape_in_trace,
                )
                if isinstance(op, Conv2dOp):
                    traced_weight = tuple(
                        [traced_weight[axis] for axis in [2, 3, 1, 0]]
                    )
                elif isinstance(op, DenseOp):
                    traced_weight = tuple([traced_weight[axis] for axis in [1, 0]])
                else:
                    raise RuntimeError()
                mask = np.ones(weight.shape, dtype=np.int32)
                mask[traced_weight] = 0
                weight[mask.astype(np.bool)] = 0
        load_variables_into_tf(tf_graph, variables, session)


class FetchInnerTensorsHook(SessionRunHook):
    def __init__(
        self, out, image: tf.Tensor, logits: tf.Tensor, merge_fn=None, graph=None
    ):
        self.out = out
        self.input_name = image.name
        self.tf_graph = logits.graph
        self.image = image
        self.logits = logits
        self.merge_fn = merge_fn or (lambda out, graph: out.append(graph))
        self.graph = graph
        # self.tr = tracker.SummaryTracker()

    def after_create_session(self, session, coord):
        if self.graph is None:
            with session.as_default():
                self.graph = build_graph(inputs=[self.image], outputs=[self.logits])
        self.tensors = {
            tensor.name: self.tf_graph.get_tensor_by_name(tensor.name)
            for tensor in self.graph.tensors
        }
        self.variables = {
            name: self.tf_graph.get_tensor_by_name(name)
            for name, _ in self.graph.variables.items()
        }

    def before_run(self, run_context):
        return SessionRunArgs(fetches={**self.tensors, **self.variables})

    def after_run(self, run_context, run_values):
        # self.tr.print_diff()
        batch_size = run_values.results[self.input_name].shape[0]
        for i in range(batch_size):
            graph = copy.deepcopy(self.graph)
            for name, variable in self.tensors.items():
                graph.tensor(graph.id(name)).value = run_values.results[name][i]
            assign_variables(graph, run_values.results)
            self.merge_fn(self.out, graph)


def model_fn_with_fetch_hook(
    features, labels, mode, params, create_model, out, merge_fn=None, graph=None
):
    image = features
    if isinstance(image, dict):
        image = features["image"]

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = create_model()(image, training=False)
        predictions = {"classes": tf.argmax(logits, axis=1)}
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            prediction_hooks=[
                FetchInnerTensorsHook(out, image, logits, merge_fn, graph)
            ],
            export_outputs={"classify": tf.estimator.export.PredictOutput(predictions)},
        )


def build_graph(inputs: List[tf.Tensor], outputs: List[tf.Tensor]) -> Graph:
    graph = Graph()
    visited_tensors = {}
    visited_ops = {}
    for input in inputs:
        iterate_tf_graph_from_tensor(graph, input, visited_tensors, visited_ops)
        graph.add_input(visited_tensors[input])
    for output in outputs:
        graph.add_output(visited_tensors[output])
    tf_graph = inputs[0].graph
    with tf_graph.as_default():
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return graph.rewrite(
        RewriteToVariable(graph, variables),
        RewriteToBatchNorm(),
        RemoveIdentity(),
        FuseConv2d(),
        FuseDense(),
        FuseBiasAdd(),
        FuseReshape(),
        FuseTranspose(),
        FusePad(),
        FuseConcatOp(),
        FuseMean(),
        RewriteToRelu(),
        RewriteToPool(),
        RewriteToAdd(),
        RemoveMulForBatchNorm(),
        RewriteToSqueezeOp(),
    )


def iterate_tf_graph_from_tensor(
    graph: Graph,
    current_tensor: tf.Tensor,
    visited_tensors: Dict[tf.Tensor, int],
    visited_ops: Dict[tf.Operation, int],
):
    if current_tensor not in visited_tensors:
        wrapped_tensor = Tensor(
            graph,
            current_tensor.name,
            shape=current_tensor.shape.as_list()
            if current_tensor.shape.dims is not None
            else None,
            dtype=current_tensor.dtype.as_numpy_dtype
            if current_tensor.dtype.is_numpy_compatible
            else None,
        )
        visited_tensors[current_tensor] = wrapped_tensor.id
        iterate_tf_graph_from_op(graph, current_tensor.op, visited_tensors, visited_ops)
        wrapped_tensor.op_id = visited_ops[current_tensor.op]
        for op in current_tensor.consumers():
            iterate_tf_graph_from_op(graph, op, visited_tensors, visited_ops)
            wrapped_tensor.add_output(visited_ops[op])


def iterate_tf_graph_from_op(
    graph: Graph,
    current_op: tf.Operation,
    visited_tensors: Dict[tf.Tensor, int],
    visited_ops: Dict[tf.Operation, int],
):
    if current_op not in visited_ops:
        wrapped_op = TensorFlowOperation(graph, current_op)
        visited_ops[current_op] = wrapped_op.id
        for tensor in current_op.inputs:
            iterate_tf_graph_from_tensor(graph, tensor, visited_tensors, visited_ops)
            wrapped_op.add_input(visited_tensors[tensor])
        for tensor in current_op.outputs:
            iterate_tf_graph_from_tensor(graph, tensor, visited_tensors, visited_ops)
            wrapped_op.add_output(visited_tensors[tensor])
