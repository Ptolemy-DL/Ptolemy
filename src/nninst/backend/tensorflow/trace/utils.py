from typing import Callable, List, Optional

import numpy as np

from nninst import Graph
from nninst.op import ReluOp
from nninst.utils.numpy import arg_approx


def get_variant(
    example_num: int = 0,
    layer_num: int = 0,
    seed_threshold: float = None,
    early_stop_layer_num: int = None,
) -> str:
    variant = []
    if example_num != 0:
        variant.append(f"n{example_num}")
    if layer_num != 0:
        variant.append(f"[layer_num={layer_num}]")
    if seed_threshold is not None:
        variant.append(f"[seed_threshold={seed_threshold:.3f}]")
    if early_stop_layer_num is not None:
        variant.append(f"[early_stop={early_stop_layer_num}]")

    if len(variant) == 0:
        return None
    else:
        return "".join(variant)


def get_entry_points(graph: Graph, layer_num: int) -> Optional[List[int]]:
    if layer_num != 0:
        # last_op = graph.ops_in_layers(Conv2dOp, DenseOp)[layer_num - 1]
        last_op = graph.ops_in_layers(ReluOp)[layer_num - 1]
        return graph.op(graph.id(last_op)).outputs
    else:
        return None


def can_support_diff(layer_num: int) -> bool:
    return layer_num == 0


def get_select_seed_fn(
    seed_threshold: float,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    if seed_threshold is not None:
        return lambda x: arg_approx(x, seed_threshold)
    else:
        return None
