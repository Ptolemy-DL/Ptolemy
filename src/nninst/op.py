from typing import List

import numpy as np

from .graph import Graph, Operation, Variable

__all__ = [
    "get_weight",
    "Conv2dOp",
    "DenseOp",
    "PoolOp",
    "MaxPoolOp",
    "AvgPoolOp",
    "ReluOp",
    "TransposeOp",
    "PadOp",
    "ReshapeOp",
    "BatchNormOp",
    "AddOp",
    "SqueezeOp",
    "ConcatOp",
    "MeanOp",
]


class DenseOp(Operation):
    def __init__(self, graph: Graph, name: str, weight: Variable, bias: Variable):
        super().__init__(graph, name)
        self.variables.extend([weight, bias])
        self.weight = weight
        self.bias = bias


class Conv2dOp(Operation):
    def __init__(
        self,
        graph: Graph,
        name: str,
        kernel: Variable,
        bias: Variable,
        padding: str,
        strides=None,
        dilations=None,
        data_format: str = None,
    ):
        super().__init__(graph, name)
        self.variables.extend([kernel, bias])
        self.kernel = kernel
        self.bias = bias
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilations
        self.data_format = data_format


def get_weight(op: Operation) -> Variable:
    if isinstance(op, DenseOp):
        return op.weight
    elif isinstance(op, Conv2dOp):
        return op.kernel
    else:
        raise RuntimeError(f"op {op.name} with type {type(op)} doesn't have weight")


class PoolOp(Operation):
    def __init__(
        self,
        graph: Graph,
        name: str,
        filter_shape: List[int],
        padding: str,
        strides: List[int] = None,
        data_format: str = None,
    ):
        super().__init__(graph, name)
        self.filter_shape = filter_shape
        self.padding = padding
        self.strides = strides
        self.data_format = data_format


class MaxPoolOp(PoolOp):
    ...


class AvgPoolOp(PoolOp):
    ...


class TransposeOp(Operation):
    def __init__(self, graph: Graph, name: str, perm: List[int]):
        super().__init__(graph, name)
        self.perm = perm


class ConcatOp(Operation):
    def __init__(self, graph: Graph, name: str, axis: int):
        super().__init__(graph, name)
        self.axis = axis


class SqueezeOp(Operation):
    def __init__(self, graph: Graph, name: str, squeeze_dims: List[int]):
        super().__init__(graph, name)
        self.squeeze_dims = squeeze_dims


class PadOp(Operation):
    def __init__(self, graph: Graph, name: str, paddings: np.ndarray):
        super().__init__(graph, name)
        self.paddings = paddings


class MeanOp(Operation):
    def __init__(self, graph: Graph, name: str, reduction_indices: List[int]):
        super().__init__(graph, name)
        self.reduction_indices = reduction_indices


class ReshapeOp(Operation):
    ...


class ReluOp(Operation):
    ...


class BatchNormOp(Operation):
    ...


class AddOp(Operation):
    ...
