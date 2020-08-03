from typing import Any, Dict, List

import numpy as np
from tensorflow import Variable as TFVariable

from nninst import Graph, Node, Tensor, Variable
from nninst.graph import Rule
from nninst.op import *

from .op import TensorFlowOperation

__all__ = [
    "FuseBiasAdd",
    "FuseConv2d",
    "FuseDense",
    "FuseReshape",
    "FuseTranspose",
    "FusePad",
    "FuseConcatOp",
    "FuseMean",
    "RewriteToPool",
    "RewriteToRelu",
    "RewriteToBatchNorm",
    "RewriteToAdd",
    "RewriteToSqueezeOp",
    "RewriteToVariable",
    "RemoveIdentity",
]


class RewriteToVariable(Rule):
    def __init__(self, graph: Graph, variables: List[TFVariable]):
        self.variable_by_node: Dict[int, Tensor] = {}
        for variable in variables:
            if variable.name == variable.op.outputs[0].name:
                node_name = variable.name
            else:
                node_name = (
                    next(
                        op
                        for op in variable.op.outputs[0].consumers()
                        if op.type == "Identity"
                    )
                    .outputs[0]
                    .name
                )
            if graph.contains_name(node_name):
                self.variable_by_node[graph.id(node_name)] = Tensor(
                    graph,
                    variable.name,
                    shape=variable.shape.as_list()
                    if variable.shape.dims is not None
                    else None,
                    dtype=variable.dtype.as_numpy_dtype
                    if variable.dtype.is_numpy_compatible
                    else None,
                )

    def action(self, node: Node) -> Any:
        if node.id in self.variable_by_node:
            variable = self.variable_by_node[node.id]
            return {node.id: {"to": variable.id, "output": variable.id}}


class RemoveIdentity(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "Identity":
            graph = node.graph
            input_tensor = graph.tensor(node.inputs[0])
            output_tensor = graph.tensor(node.outputs[0])
            if len(input_tensor.inputs) != 0:
                op_id = input_tensor.op_id
                return {node.id: {"to": op_id, "output": op_id}}
            else:
                assert len(output_tensor.outputs) == 1
                return {
                    node.id: {"input": output_tensor.outputs[0]},
                    "next": input_tensor.id,
                }


class FuseBiasAdd(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "BiasAdd":
            graph = node.graph
            input_tensors = [graph.tensor(tensor_id) for tensor_id in node.inputs]
            bias = None
            input_tensor = None
            if (
                len(input_tensors[0].inputs) == 0
                or (
                    isinstance(input_tensors[0].input_nodes[0], TensorFlowOperation)
                    and input_tensors[0].input_nodes[0].tf_op.type
                    in ["ReadVariableOp", "VariableV2"]
                )
            ) and input_tensors[0].id not in graph.inputs:
                bias = input_tensors[0]
                input_tensor = input_tensors[1]
            elif (
                len(input_tensors[1].inputs) == 0
                or (
                    isinstance(input_tensors[1].input_nodes[0], TensorFlowOperation)
                    and input_tensors[1].input_nodes[0].tf_op.type
                    in ["ReadVariableOp", "VariableV2"]
                )
            ) and input_tensors[1].id not in graph.inputs:
                bias = input_tensors[1]
                input_tensor = input_tensors[0]
            op = input_tensor.op
            if bias is not None and op is not None:
                if isinstance(op, (Conv2dOp, DenseOp)):
                    op.bias = Variable(bias.name)
                    op.variables.append(op.bias)
                    return {node.id: {"to": op.id, "output": op.id}}


class FuseConv2d(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "Conv2D":
            graph = node.graph
            input_tensors = [graph.tensor(tensor_id) for tensor_id in node.inputs]
            kernel = None
            upstream_index = None
            if (
                len(input_tensors[0].inputs) == 0
                or (
                    isinstance(input_tensors[0].input_nodes[0], TensorFlowOperation)
                    and input_tensors[0].input_nodes[0].tf_op.type
                    in ["ReadVariableOp", "VariableV2"]
                )
            ) and input_tensors[0].id not in graph.inputs:
                kernel = input_tensors[0]
                upstream_index = 1
            elif (
                len(input_tensors[1].inputs) == 0
                or (
                    isinstance(input_tensors[1].input_nodes[0], TensorFlowOperation)
                    and input_tensors[1].input_nodes[0].tf_op.type
                    in ["ReadVariableOp", "VariableV2"]
                )
            ) and input_tensors[1].id not in graph.inputs:
                kernel = input_tensors[1]
                upstream_index = 0
            if kernel is not None:
                tf_conv2d_def = node.tf_op.node_def
                data_format = tf_conv2d_def.attr["data_format"].s.decode()
                strides = list(tf_conv2d_def.attr["strides"].list.i)
                if data_format == "NHWC":
                    strides = strides[1:3]
                elif data_format == "NCHW":
                    strides = strides[2:]
                else:
                    raise RuntimeError(
                        f"{data_format} is not supported value of data format"
                    )
                op = Conv2dOp(
                    graph,
                    node.name,
                    kernel=Variable(kernel.name),
                    bias=None,
                    padding=tf_conv2d_def.attr["padding"].s.decode(),
                    strides=strides,
                    dilations=list(tf_conv2d_def.attr["dilations"].list.i),
                    data_format=data_format,
                )
                return {
                    node.id: {
                        "to": op.id,
                        "input": {upstream_index: op.id},
                        "output": op.id,
                    }
                }


class FuseDense(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "MatMul":
            graph = node.graph
            input_tensors = [graph.tensor(tensor_id) for tensor_id in node.inputs]
            weight = None
            upstream_index = None
            if (
                len(input_tensors[0].inputs) == 0
                or (
                    isinstance(input_tensors[0].input_nodes[0], TensorFlowOperation)
                    and input_tensors[0].input_nodes[0].tf_op.type
                    in ["ReadVariableOp", "VariableV2"]
                )
            ) and input_tensors[0].id not in graph.inputs:
                weight = input_tensors[0]
                upstream_index = 1
            elif (
                len(input_tensors[1].inputs) == 0
                or (
                    isinstance(input_tensors[1].input_nodes[0], TensorFlowOperation)
                    and input_tensors[1].input_nodes[0].tf_op.type
                    in ["ReadVariableOp", "VariableV2"]
                )
            ) and input_tensors[1].id not in graph.inputs:
                weight = input_tensors[1]
                upstream_index = 0
            if weight is not None:
                op = DenseOp(graph, node.name, weight=Variable(weight.name), bias=None)
                return {
                    node.id: {
                        "to": op.id,
                        "input": {upstream_index: op.id},
                        "output": op.id,
                    }
                }


class FuseReshape(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "Reshape":
            graph = node.graph
            op = ReshapeOp(graph, node.name)
            input_index = list(
                filter(
                    lambda input_index: not (
                        graph.tensor(node.inputs[input_index]).dtype == np.int32
                        and len(graph.tensor(node.inputs[input_index]).shape) == 1
                    ),
                    range(len(node.inputs)),
                )
            )[0]
            return {
                node.id: {"to": op.id, "output": op.id, "input": {input_index: op.id}}
            }


class FuseTranspose(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "Transpose":
            graph = node.graph
            input_index = list(
                filter(
                    lambda input_index: not (
                        graph.tensor(node.inputs[input_index]).dtype == np.int32
                        and len(graph.tensor(node.inputs[input_index]).shape) == 1
                    ),
                    range(len(node.inputs)),
                )
            )[0]
            perm_index = 0 if input_index == 1 else 1
            op = TransposeOp(
                graph,
                node.name,
                perm=list(
                    graph.tensor(node.inputs[perm_index]).op.tf_op.outputs[0].eval()
                ),
            )
            return {
                node.id: {"to": op.id, "output": op.id, "input": {input_index: op.id}}
            }


class FuseConcatOp(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "ConcatV2":
            graph = node.graph
            axis_index = list(
                filter(
                    lambda input_index: graph.tensor(node.inputs[input_index]).dtype
                    == np.int32
                    and len(graph.tensor(node.inputs[input_index]).shape) == 0,
                    range(len(node.inputs)),
                )
            )[0]
            input_indices = list(
                filter(
                    lambda input_index: input_index != axis_index,
                    range(len(node.inputs)),
                )
            )
            op = ConcatOp(
                graph,
                node.name,
                axis=graph.tensor(node.inputs[axis_index]).op.tf_op.outputs[0].eval(),
            )
            return {
                node.id: {
                    "to": op.id,
                    "output": op.id,
                    "input": {input_index: op.id for input_index in input_indices},
                }
            }


class FusePad(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "Pad":
            graph = node.graph
            input_index = list(
                filter(
                    lambda input_index: not (
                        graph.tensor(node.inputs[input_index]).dtype == np.int32
                    ),
                    range(len(node.inputs)),
                )
            )[0]
            paddings_index = 0 if input_index == 1 else 1
            op = PadOp(
                graph,
                node.name,
                paddings=graph.tensor(node.inputs[paddings_index])
                .op.tf_op.outputs[0]
                .eval(),
            )
            return {
                node.id: {"to": op.id, "output": op.id, "input": {input_index: op.id}}
            }


class RewriteToRelu(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "Relu":
            graph = node.graph
            op = ReluOp(graph, node.name)
            return op.id


class RewriteToPool(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type in [
            "MaxPool",
            "AvgPool",
        ]:
            graph = node.graph
            tf_pool_def = node.tf_op.node_def
            data_format = tf_pool_def.attr["data_format"].s.decode()
            filter_shape = list(tf_pool_def.attr["ksize"].list.i)
            strides = list(tf_pool_def.attr["strides"].list.i)
            if data_format == "NHWC":
                filter_shape = filter_shape[1:3]
                strides = strides[1:3]
            elif data_format == "NCHW":
                filter_shape = filter_shape[2:]
                strides = strides[2:]
            else:
                raise RuntimeError(
                    f"{data_format} is not supported value of data format"
                )
            padding = tf_pool_def.attr["padding"].s.decode()
            if node.tf_op.type == "MaxPool":
                op = MaxPoolOp(
                    graph,
                    node.name,
                    filter_shape=filter_shape,
                    padding=padding,
                    strides=strides,
                    data_format=data_format,
                )
            elif node.tf_op.type == "AvgPool":
                op = AvgPoolOp(
                    graph,
                    node.name,
                    filter_shape=filter_shape,
                    padding=padding,
                    strides=strides,
                    data_format=data_format,
                )
            return op.id


class RewriteToBatchNorm(Rule):
    def action(self, node: Node) -> Any:
        if (
            isinstance(node, TensorFlowOperation)
            and node.tf_op.type == "FusedBatchNorm"
        ):
            graph = node.graph
            op = BatchNormOp(graph, node.name)
            input_index = list(
                filter(
                    lambda input_index: len(
                        graph.tensor(node.inputs[input_index]).shape
                    )
                    > 1,
                    range(len(node.inputs)),
                )
            )[0]
            return {
                node.id: {"to": op.id, "output": op.id, "input": {input_index: op.id}}
            }


class RewriteToSqueezeOp(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "Squeeze":
            graph = node.graph
            tf_pool_def = node.tf_op.node_def
            squeeze_dims = list(tf_pool_def.attr["squeeze_dims"].list.i)
            op = SqueezeOp(graph, node.name, squeeze_dims=squeeze_dims)
            return op.id


class RewriteToAdd(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "Add":
            graph = node.graph
            if len(node.inputs) == 2 and (
                (isinstance(node.input_nodes[0].input_nodes[0], TensorFlowOperation))
                and isinstance(node.input_nodes[0].input_nodes[0], TensorFlowOperation)
                and (
                    (
                        node.input_nodes[0].input_nodes[0].tf_op.type == "Mul"
                        and node.input_nodes[1].input_nodes[0].tf_op.type == "Sub"
                    )
                    or (
                        node.input_nodes[1].input_nodes[0].tf_op.type == "Mul"
                        and node.input_nodes[0].input_nodes[0].tf_op.type == "Sub"
                    )
                )
            ):
                op = BatchNormOp(graph, node.name)
                input_index = list(
                    filter(
                        lambda input_index: len(
                            graph.tensor(node.inputs[input_index]).shape
                        )
                        > 1,
                        range(len(node.inputs)),
                    )
                )[0]
                return {
                    node.id: {
                        "to": op.id,
                        "output": op.id,
                        "input": {input_index: op.id},
                    }
                }
            else:
                op = AddOp(graph, node.name)
                return op.id


class FuseMean(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "Mean":
            graph = node.graph
            input_index = list(
                filter(
                    lambda input_index: not (
                        graph.tensor(node.inputs[input_index]).dtype == np.int32
                    ),
                    range(len(node.inputs)),
                )
            )[0]
            reduction_indices_index = 0 if input_index == 1 else 1
            op = MeanOp(
                graph,
                node.name,
                reduction_indices=list(
                    graph.tensor(node.inputs[reduction_indices_index])
                    .op.tf_op.outputs[0]
                    .eval()
                ),
            )
            return {
                node.id: {"to": op.id, "output": op.id, "input": {input_index: op.id}}
            }


class RemoveMulForBatchNorm(Rule):
    def action(self, node: Node) -> Any:
        if isinstance(node, TensorFlowOperation) and node.tf_op.type == "Mul":
            graph = node.graph
            output_tensor = graph.tensor(node.outputs[0])
            if isinstance(output_tensor.output_nodes[0], BatchNormOp):
                input_index = list(
                    filter(
                        lambda input_index: len(
                            graph.tensor(node.inputs[input_index]).shape
                        )
                        > 1,
                        range(len(node.inputs)),
                    )
                )[0]
                input_tensor = graph.tensor(node.inputs[input_index])
                op_id = input_tensor.op_id
                return {node.id: {"to": op_id, "output": op_id}}
