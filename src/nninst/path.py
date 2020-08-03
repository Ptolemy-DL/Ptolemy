import operator
from functools import reduce
from typing import Callable, Dict, Type, Union

import numpy as np
import pandas as pd

from .graph import AttrMap, Graph, Operation, Tensor
from .op import *
from .trace import TraceKey, calc_padding


def op_is_in_fc_layers(op: Operation) -> bool:
    return isinstance(op, (DenseOp, ReluOp))


def get_trace_path_intersection_in_fc_layers(
    *traces: AttrMap, graph: Graph, compact: bool = False
) -> AttrMap:
    return get_trace_path_intersection(
        *traces, graph=graph, filter_func=op_is_in_fc_layers, compact=compact
    )


def get_trace_path_intersection(
    *traces: AttrMap,
    graph: Graph,
    filter_func: Callable[[Operation], bool] = None,
    compact: bool = False,
) -> AttrMap:
    first_trace = traces[0]

    def set_output_point(tensor: Tensor):
        assert np.all(
            reduce(
                operator.eq,
                [trace.tensors[tensor.name][TraceKey.POINT] for trace in traces],
            )
        )
        tensor.attrs[TraceKey.POINT] = first_trace.tensors[tensor.name][TraceKey.POINT]

    def set_edge_intersection(op: Operation):
        if TraceKey.is_trivial(op):
            return
        edges = [trace.ops[op.name][TraceKey.EDGE] for trace in traces]
        if compact:
            edge_intersection = reduce(np.bitwise_and, edges)
        else:
            edge_intersection = reduce(np.intersect1d, map(TraceKey.to_array, edges))
        op.attrs[TraceKey.EDGE] = edge_intersection

    new_graph = graph.with_attrs(TraceKey.filter_key(TraceKey.META, first_trace))
    reconstruct_trace_path_with_hook(
        new_graph,
        on_enter_output_tensor=set_output_point,
        on_enter_op=set_edge_intersection,
        filter_func=filter_func,
        compact=compact,
    )
    return new_graph.attrs_to_map()


_trace_path_func_by_op: Dict[Type[Operation], Callable[..., None]] = {}


def register_op(op_type: Type[Operation], trace_func: Callable[..., None]):
    _trace_path_func_by_op[op_type] = trace_func


def get_trace_path_in_fc_layers(
    graph: Graph, trace: AttrMap, compact: bool = False
) -> AttrMap:
    return get_trace_path(
        graph=graph, trace=trace, filter_func=op_is_in_fc_layers, compact=compact
    )


def get_trace_path(
    graph: Graph,
    trace: AttrMap,
    filter_func: Callable[[Operation], bool] = None,
    compact: bool = False,
) -> AttrMap:
    graph_with_trace = graph.with_attrs(trace)
    reconstruct_trace_path_with_hook(
        graph_with_trace,
        on_enter_output_tensor=lambda _: None,
        on_enter_op=lambda _: None,
        filter_func=filter_func,
        compact=compact,
    )
    return TraceKey.filter_key(
        TraceKey.META | {TraceKey.PATH}, graph_with_trace.attrs_to_map()
    )


def reconstruct_trace_path_with_hook(
    graph: Graph,
    on_enter_output_tensor: Callable[[Tensor], None],
    on_enter_op: Callable[[Operation], None],
    filter_func: Callable[[Operation], bool] = None,
    compact: bool = False,
):
    op_to_wait_count = {op.id: len(op.outputs) for op in graph.ops}
    tensor_to_wait_count = {tensor.id: len(tensor.outputs) for tensor in graph.tensors}
    for output_id in graph.outputs:
        output_tensor = graph.tensor(output_id)
        on_enter_output_tensor(output_tensor)
        output_points = output_tensor.attrs[TraceKey.POINT]
        output_tensor.attrs[TraceKey.PATH] = TraceKey.to_frame(output_points, compact)
    ready_ops = list([graph.tensor(output_id).op_id for output_id in graph.outputs])
    while len(ready_ops) != 0:
        ready_op_id = ready_ops.pop()
        ready_op = graph.op(ready_op_id)
        if filter_func is None or filter_func(ready_op):
            on_enter_op(ready_op)
            _trace_path_func_by_op[type(ready_op)](ready_op, compact)
            for input_tensor_id in ready_op.inputs:
                tensor_to_wait_count[input_tensor_id] = (
                    tensor_to_wait_count[input_tensor_id] - 1
                )
                if tensor_to_wait_count[input_tensor_id] == 0:
                    tensor_to_wait_count.pop(input_tensor_id)
                    input_tensor = graph.tensor(input_tensor_id)
                    if input_tensor.op is not None:
                        input_op_id = input_tensor.op.id
                        op_to_wait_count[input_op_id] = (
                            op_to_wait_count[input_op_id] - 1
                        )
                        if op_to_wait_count[input_op_id] == 0:
                            op_to_wait_count.pop(input_op_id)
                            ready_ops.append(input_op_id)


def set_input_path(op: Operation, edge_mask: np.ndarray):
    output_tensor: Tensor = op.output_nodes[0]
    output_path: pd.DataFrame = output_tensor.attrs[TraceKey.PATH]
    output_point_mask = np.zeros(
        np.prod(output_tensor.attrs[TraceKey.POINT_SHAPE]), dtype=np.int8
    )
    output_point_mask[output_path.index.values] = output_path["count"]
    input_point_mask = np.dot(edge_mask, output_point_mask)
    input_point_index = np.nonzero(input_point_mask)
    input_tensor: Tensor = op.input_nodes[0]
    input_tensor.attrs[TraceKey.PATH] = pd.DataFrame(
        dict(count=input_point_mask[input_point_index]), index=input_point_index[0]
    )


def linear_layer_trace(op: DenseOp, compact: bool, *args, **kwargs):
    edge_mask = TraceKey.to_mask(
        op.attrs[TraceKey.EDGE], op.attrs[TraceKey.EDGE_SHAPE], compact
    )
    set_input_path(op, edge_mask)


register_op(DenseOp, linear_layer_trace)


def get_edge_mask(op: Union[Conv2dOp, PoolOp], compact: bool) -> np.ndarray:
    edge_shape = op.attrs[TraceKey.EDGE_SHAPE]
    if not compact:
        return TraceKey.to_mask(
            op.attrs[TraceKey.EDGE], (np.prod(edge_shape[:3], edge_shape[3:])), compact
        )
    input_tensor: Tensor = op.input_nodes[0]
    output_tensor: Tensor = op.output_nodes[0]
    edge = TraceKey.to_array(op.attrs[TraceKey.EDGE], compact)
    input_shape = input_tensor.attrs[TraceKey.POINT_SHAPE]
    output_shape = output_tensor.attrs[TraceKey.POINT_SHAPE]
    if op.data_format == "NHWC":
        input_shape = (input_shape[2], input_shape[0], input_shape[1])
        output_shape = (output_shape[2], output_shape[0], output_shape[1])
    if isinstance(op, Conv2dOp):
        in_channel, kernel_height, kernel_width, out_channel, out_height, out_width = np.unravel_index(
            edge, edge_shape
        )
    else:
        kernel_height, kernel_width, out_channel, out_height, out_width = np.unravel_index(
            edge, edge_shape
        )
    stride = np.array(op.strides)
    kernel_size = (
        np.array(op.attrs[TraceKey.WEIGHT_SHAPE])[2:]
        if isinstance(op, Conv2dOp)
        else np.array(op.filter_shape)
    )
    padding = calc_padding(
        np.array(input_shape)[1:], np.array(output_shape)[1:], stride, kernel_size
    )
    in_height = kernel_height + (out_height * stride[0]) - padding[1][0]
    in_width = kernel_width + (out_width * stride[1]) - padding[2][0]
    edge_output_index = np.ravel_multi_index(
        (out_channel, out_height, out_width), output_shape
    )
    if isinstance(op, Conv2dOp):
        edge_input_index = np.ravel_multi_index(
            (in_channel, in_height, in_width), input_shape
        )
    else:
        edge_input_index = np.ravel_multi_index(
            (out_channel, in_height, in_width), input_shape
        )
    mask = np.zeros((np.prod(input_shape), np.prod(output_shape)), dtype=np.int8)
    mask[(edge_input_index, edge_output_index)] = 1
    return mask


def conv_layer_trace(op: Union[Conv2dOp, PoolOp], compact: bool, *args, **kwargs):
    edge_mask = get_edge_mask(op, compact)
    set_input_path(op, edge_mask)


register_op(Conv2dOp, conv_layer_trace)
register_op(AvgPoolOp, conv_layer_trace)
register_op(MaxPoolOp, conv_layer_trace)


def trivial_layer_trace(op, *args, **kwargs):
    input_tensor: Tensor = op.input_nodes[0]
    output_tensor: Tensor = op.output_nodes[0]
    input_tensor.attrs[TraceKey.PATH] = output_tensor.attrs[TraceKey.PATH]


register_op(ReluOp, trivial_layer_trace)
register_op(ReshapeOp, trivial_layer_trace)
register_op(SqueezeOp, trivial_layer_trace)
register_op(BatchNormOp, trivial_layer_trace)
