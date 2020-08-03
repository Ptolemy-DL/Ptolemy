from typing import Callable, Dict, List, Type

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional

from . import mode
from .graph import AttrMap, Graph, Operation, Tensor
from .op import *
from .trace import TraceKey, calc_padding
from .utils.numpy import argtopk, concatenate, repeat

__all__ = ["get_channel_trace", "reconstruct_channel_trace"]

_trace_func_by_op: Dict[Type[Operation], Callable[..., None]] = {}

EPS = np.finfo(np.float16).eps
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def register_op(op_type: Type[Operation], trace_func: Callable[..., None]):
    _trace_func_by_op[op_type] = trace_func


def get_channel_trace(
    graph: Graph,
    select_fn: Callable[[np.ndarray], np.ndarray],
    select_seed_fn: Callable[[np.ndarray], np.ndarray] = None,
    entry_points: List[int] = None,
    debug: bool = False,
    *args,
    **kwargs,
) -> AttrMap:
    new_graph = graph.clone()
    reconstruct_channel_trace(
        graph=new_graph,
        select_fn=select_fn,
        select_seed_fn=select_seed_fn,
        entry_points=entry_points,
        debug=debug,
        *args,
        **kwargs,
    )
    return new_graph.attrs_to_map()


def reconstruct_channel_trace(
    graph: Graph,
    select_fn: Callable[[np.ndarray], np.ndarray],
    select_seed_fn: Callable[[np.ndarray], np.ndarray] = None,
    entry_points: List[int] = None,
    debug: bool = False,
    *args,
    **kwargs,
):
    assert TraceKey.DATA_FORMAT not in graph.attrs
    select_seed_fn = select_seed_fn or (lambda output: argtopk(output, 1))
    entry_points = entry_points or graph.outputs
    data_formats = np.array(
        [op.data_format for op in graph.ops if isinstance(op, (Conv2dOp, PoolOp))]
    )
    if np.all(data_formats == "NHWC"):
        graph.attrs[TraceKey.DATA_FORMAT] = "NHWC"
    elif np.all(data_formats == "NCHW"):
        graph.attrs[TraceKey.DATA_FORMAT] = "NCHW"
    else:
        raise RuntimeError("data format in graph is inconsistent")
    if graph.attrs[TraceKey.DATA_FORMAT] == "NHWC":
        for tensor in graph.tensors:
            tensor.value = np.rollaxis(tensor.value, 2)
    op_to_wait_count = {op.id: len(op.outputs) for op in graph.ops}
    tensor_to_wait_count = {tensor.id: len(tensor.outputs) for tensor in graph.tensors}
    for output_id in entry_points:
        output_tensor = graph.tensor(output_id)
        output_tensor.attrs[TraceKey.POINT] = select_seed_fn(output_tensor.value)
        output_tensor.attrs[TraceKey.POINT_SHAPE] = output_tensor.value.shape
    ready_ops = list([graph.tensor(output_id).op_id for output_id in entry_points])
    while len(ready_ops) != 0:
        ready_op_id = ready_ops.pop()
        ready_op = graph.op(ready_op_id)
        ready_op.attrs[TraceKey.OP_TYPE] = type(ready_op).__name__
        _trace_func_by_op[type(ready_op)](
            ready_op, select_fn=select_fn, debug=debug, *args, **kwargs
        )
        for input_tensor_id in ready_op.inputs:
            tensor_to_wait_count[input_tensor_id] = (
                tensor_to_wait_count[input_tensor_id] - 1
            )
            if tensor_to_wait_count[input_tensor_id] == 0:
                tensor_to_wait_count.pop(input_tensor_id)
                input_tensor = graph.tensor(input_tensor_id)
                if input_tensor.op is not None:
                    input_op_id = input_tensor.op.id
                    op_to_wait_count[input_op_id] = op_to_wait_count[input_op_id] - 1
                    if op_to_wait_count[input_op_id] == 0:
                        op_to_wait_count.pop(input_op_id)
                        ready_ops.append(input_op_id)
    graph.attrs[TraceKey.COUNT] = 1
    graph.attrs[TraceKey.WEIGHT_NUM] = sum(
        op.attrs[TraceKey.WEIGHT].size
        for op in graph.ops
        if TraceKey.WEIGHT in op.attrs
    )
    graph.attrs[TraceKey.EDGE_NUM] = sum(
        op.attrs[TraceKey.EDGE].size for op in graph.ops if TraceKey.EDGE in op.attrs
    )
    graph.attrs[TraceKey.POINT_NUM] = sum(
        tensor.attrs[TraceKey.POINT].size
        for tensor in graph.tensors
        if TraceKey.POINT in tensor.attrs
    )
    graph.attrs[TraceKey.MAX_WEIGHT_NUM] = graph.attrs[TraceKey.WEIGHT_NUM]
    graph.attrs[TraceKey.MAX_POINT_NUM] = graph.attrs[TraceKey.POINT_NUM]
    graph.attrs[TraceKey.MAX_EDGE_NUM] = graph.attrs[TraceKey.EDGE_NUM]
    graph.attrs[TraceKey.MIN_WEIGHT_NUM] = graph.attrs[TraceKey.WEIGHT_NUM]
    graph.attrs[TraceKey.MIN_POINT_NUM] = graph.attrs[TraceKey.POINT_NUM]
    graph.attrs[TraceKey.MIN_EDGE_NUM] = graph.attrs[TraceKey.EDGE_NUM]


def merge_traced_points(
    tensor: Tensor, op: Operation, traced_points: np.ndarray, is_unique: bool = False
):
    tensor.attrs[TraceKey.POINT_SHAPE] = (tensor.value.shape[0],)
    if not is_unique:
        traced_points = np.unique(traced_points)
    op_index = tensor.outputs.index(op.id)
    tensor.attrs[TraceKey.POINT + f".{op_index}"] = traced_points
    if TraceKey.POINT in tensor.attrs:
        tensor.attrs[TraceKey.POINT] = np.unique(
            concatenate([traced_points, tensor.attrs[TraceKey.POINT]], dtype=np.int32)
        )
    else:
        tensor.attrs[TraceKey.POINT] = traced_points


def select_input(output_value, weighted_input, select_fn):
    if output_value < 0:
        flipped_weighed_input = -weighted_input
    else:
        flipped_weighed_input = weighted_input
    input_points = select_fn(flipped_weighed_input)
    return input_points


def linear_layer_trace(
    op: DenseOp, select_fn: Callable[[np.ndarray], np.ndarray], *args, **kwargs
):
    weight = op.weight.value
    input_tensor: Tensor = op.input_nodes[0]
    input = input_tensor.value
    output_tensor: Tensor = op.output_nodes[0]
    output: np.ndarray = output_tensor.value
    output_trace_points = []
    input_trace_points = []
    for index, output_point in enumerate(output_tensor.attrs[TraceKey.POINT]):
        weighted_input = weight[output_point] * input
        output_value = output[output_point]
        if mode.is_check():
            if op.bias is not None:
                weighted_input_sum = (
                    np.sum(weighted_input) + op.bias.value[output_point]
                )
            else:
                weighted_input_sum = np.sum(weighted_input)
            assert abs(output_value - weighted_input_sum) < EPS * weighted_input.size
        input_points = select_input(
            output_value=output_value,
            weighted_input=weighted_input,
            select_fn=select_fn,
        )
        output_trace_points.append(repeat(output_point, input_points.size))
        input_trace_points.append(input_points)
    output_trace_points = concatenate(output_trace_points, dtype=np.int32)
    input_trace_points = concatenate(input_trace_points, dtype=np.int32)
    edge_shape = (input.size, output.size)
    op.attrs[TraceKey.EDGE] = np.ravel_multi_index(
        (input_trace_points, output_trace_points), edge_shape
    )
    op.attrs[TraceKey.EDGE_SHAPE] = edge_shape
    merge_traced_points(input_tensor, op, input_trace_points)


register_op(DenseOp, linear_layer_trace)


def conv2d_layer_trace(
    op: Conv2dOp, select_fn: Callable[[np.ndarray], np.ndarray], *args, **kwargs
):
    weight: np.ndarray = op.kernel.value
    input_tensor: Tensor = op.input_nodes[0]
    input = input_tensor.value
    output_tensor: Tensor = op.output_nodes[0]
    output = output_tensor.value
    output_points = output_tensor.attrs[TraceKey.POINT]

    kernel_size = np.array(weight.shape)[2:]
    stride = np.array(op.strides)
    padding = calc_padding(
        np.array(input.shape)[1:], np.array(output.shape)[1:], stride, kernel_size
    )
    input = np.pad(input, padding, mode="constant")

    output_trace_points = []
    input_trace_points = []
    for output_point in output_points:
        if op.bias is not None:
            output_value = output[output_point] - op.bias.value[output_point]
        else:
            output_value = output[output_point]
        # kernels = weight[output_point, :, ::-1, ::-1]
        with torch.no_grad():
            weighted_inputs = [
                functional.conv2d(
                    input=Variable(
                        torch.from_numpy(
                            input[input_channel][np.newaxis, np.newaxis]
                        ).to(device)
                    ),
                    weight=Variable(
                        torch.from_numpy(
                            weight[output_point, input_channel][np.newaxis, np.newaxis]
                        ).to(device)
                    ),
                    stride=tuple(op.strides),
                )[0, 0]
                .cpu()
                .numpy()
                # signal.convolve2d(input[input_channel], kernels[input_channel], mode=op.padding.lower())
                for input_channel in range(input.shape[0])
            ]
        weighted_input_projections = np.array(
            [
                np.vdot(weighted_input, output_value)
                for weighted_input in weighted_inputs
            ]
        )
        if mode.is_check():
            assert (
                abs(np.linalg.norm(output_value) ** 2 - sum(weighted_input_projections))
                < EPS * input.size
            )
        input_points = select_fn(weighted_input_projections)
        repeated_output = repeat(output_point, input_points.size)
        output_trace_points.append(repeated_output)
        input_trace_points.append(input_points)
    output_trace_points = concatenate(output_trace_points, dtype=np.int32)
    input_trace_points = concatenate(input_trace_points, dtype=np.int32)
    edge_shape = (input.shape[0], output.shape[0])
    op.attrs[TraceKey.EDGE] = np.ravel_multi_index(
        (input_trace_points, output_trace_points), edge_shape
    )
    op.attrs[TraceKey.EDGE_SHAPE] = edge_shape
    merge_traced_points(input_tensor, op, input_trace_points)


register_op(Conv2dOp, conv2d_layer_trace)


def add_layer_trace(
    op: AddOp, select_fn: Callable[[np.ndarray], np.ndarray], *args, **kwargs
):
    left_input_tensor: Tensor = op.input_nodes[0]
    left_input: np.ndarray = left_input_tensor.value
    right_input_tensor: Tensor = op.input_nodes[1]
    right_input: np.ndarray = right_input_tensor.value
    output_tensor: Tensor = op.output_nodes[0]
    output: np.ndarray = output_tensor.value
    output_points = output_tensor.attrs[TraceKey.POINT]
    left_input_trace_points = []
    right_input_trace_points = []
    output_size = output.shape[0]
    for output_point in output_points:
        output_value = output[output_point]
        left_input_projection = np.vdot(left_input[output_point], output_value)
        right_input_projection = np.vdot(right_input[output_point], output_value)
        if mode.is_check():
            assert (
                abs(
                    np.linalg.norm(output_value) ** 2
                    - left_input_projection
                    - right_input_projection
                )
                < EPS * output_value.size * 2
            )
        input_points = select_fn(
            np.array([left_input_projection, right_input_projection])
        )
        if input_points.size == 1:
            if input_points[0] == 0:
                left_input_trace_points.append(output_point)
            else:
                right_input_trace_points.append(output_point)
        else:
            left_input_trace_points.append(output_point)
            right_input_trace_points.append(output_point)
    left_input_trace_points = np.array(left_input_trace_points, dtype=np.int32)
    right_input_trace_points = np.array(right_input_trace_points, dtype=np.int32)
    op.attrs[TraceKey.EDGE] = np.concatenate(
        [left_input_trace_points, right_input_trace_points + output_size]
    )
    op.attrs[TraceKey.EDGE_SHAPE] = (2, output_size)
    merge_traced_points(left_input_tensor, op, left_input_trace_points, is_unique=True)
    merge_traced_points(
        right_input_tensor, op, right_input_trace_points, is_unique=True
    )


register_op(AddOp, add_layer_trace)


def concat_layer_trace(op: ConcatOp, *args, **kwargs):
    input_tensors: List[Tensor] = op.input_nodes
    output_tensor: Tensor = op.output_nodes[0]
    output_points = output_tensor.attrs[TraceKey.POINT]
    start_index = 0
    for input_tensor in input_tensors:
        input_shape = input_tensor.value.shape[0]
        end_index = start_index + input_shape
        input_filter = np.logical_and(
            output_points >= start_index, output_points < end_index
        )
        input_points = output_points[input_filter]
        input_points = input_points - start_index
        merge_traced_points(input_tensor, op, input_points, is_unique=True)
        start_index = end_index
    op.attrs[TraceKey.TRIVIAL] = True


register_op(ConcatOp, concat_layer_trace)


def reshape_layer_trace(op: ReshapeOp, *args, **kwargs):
    input_tensor: Tensor = op.input_nodes[0]
    output_tensor: Tensor = op.output_nodes[0]
    output_points = output_tensor.attrs[TraceKey.POINT]
    input_shape = input_tensor.value.shape
    output_shape = output_tensor.value.shape
    assert len(input_shape) == 3 and len(output_shape) == 1
    op.attrs[TraceKey.TRIVIAL] = True
    merge_traced_points(
        input_tensor, op, np.unravel_index(output_points, input_shape)[0]
    )


register_op(ReshapeOp, reshape_layer_trace)


def squeeze_layer_trace(op: SqueezeOp, *args, **kwargs):
    raise NotImplementedError()


register_op(SqueezeOp, squeeze_layer_trace)


def trivial_layer_trace(op, *args, **kwargs):
    input_tensor: Tensor = op.input_nodes[0]
    output_tensor: Tensor = op.output_nodes[0]
    merge_traced_points(
        input_tensor, op, output_tensor.attrs[TraceKey.POINT], is_unique=True
    )
    op.attrs[TraceKey.TRIVIAL] = True


register_op(MaxPoolOp, trivial_layer_trace)
register_op(AvgPoolOp, trivial_layer_trace)
register_op(ReluOp, trivial_layer_trace)
# register_op(SqueezeOp, trivial_layer_trace)
register_op(BatchNormOp, trivial_layer_trace)
register_op(PadOp, trivial_layer_trace)
register_op(TransposeOp, trivial_layer_trace)
