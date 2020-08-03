import gc
from functools import reduce
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd

from .graph import AttrMap, Graph
from .trace import AddOp, TraceKey
from .utils import filter_not_null
from .utils.fs import IOAction
from .utils.ray import ray_iter

__all__ = [
    "calc_iou",
    "calc_iou_compact",
    "calc_trace_side_overlap",
    "calc_weighted_iou",
    "calc_class_trace_side_overlap",
    "calc_class_trace_side_overlap_norm",
    "calc_trace_side_overlap_compact",
    "calc_class_trace_side_overlap_compact",
    "calc_trace_side_overlap_both_compact",
    "calc_density",
    "calc_density_compact",
    "calc_space",
    "calc_skip_ratio",
    "calc_trace_size",
    "calc_density_compact_per_layer",
    "self_similarity_matrix",
    "self_similarity_matrix_ray",
]

T = TypeVar("T")


def calc_iou(trace1: AttrMap, trace2: AttrMap, key: str = TraceKey.EDGE) -> float:
    def intersect_and_union(node_name: str) -> Optional[Tuple[int, int]]:
        node_trace1 = trace1.nodes[node_name]
        if key in node_trace1:
            node_trace2 = trace2.nodes[node_name]
            trace_set1 = TraceKey.to_array(node_trace1[key])
            trace_set2 = TraceKey.to_array(node_trace2[key])
            intersect = np.intersect1d(trace_set1, trace_set2)
            union = np.union1d(trace_set1, trace_set2)
            return len(intersect), len(union)
        else:
            return None

    def get_iou(args: Tuple[int, int]) -> float:
        intersect_size, union_size = args
        return intersect_size / union_size

    iou = get_iou(
        reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1]),
            filter_not_null(
                [intersect_and_union(node_name) for node_name in trace1.nodes]
            ),
        )
    )
    return iou


def calc_iou_frequency(
    trace1: AttrMap, trace2: AttrMap, frequency: int, key: str = TraceKey.EDGE
) -> float:
    def intersect_and_union(node_name: str) -> Optional[Tuple[int, int]]:
        node_trace1 = trace1.nodes[node_name]
        if key in node_trace1:
            node_trace2 = trace2.nodes[node_name]
            trace_set1 = (
                node_trace1[key].index[node_trace1[key]["count"] > frequency].values
            )
            trace_set2 = (
                node_trace2[key].index[node_trace2[key]["count"] > frequency].values
            )
            intersect = np.intersect1d(trace_set1, trace_set2)
            union = np.union1d(trace_set1, trace_set2)
            return len(intersect), len(union)
        else:
            return None

    def get_iou(args: Tuple[int, int]) -> float:
        intersect_size, union_size = args
        if union_size == 0:
            return 0
        else:
            return intersect_size / union_size

    iou = get_iou(
        reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1]),
            filter_not_null(
                [intersect_and_union(node_name) for node_name in trace1.nodes]
            ),
        )
    )
    return iou


def calc_iou_frequency_per_layer(
    trace1: AttrMap,
    trace2: AttrMap,
    node_name: str,
    frequency: int,
    key: str = TraceKey.EDGE,
) -> float:
    node_trace1 = trace1.nodes[node_name]
    if key in node_trace1:
        node_trace2 = trace2.nodes[node_name]
        trace_set1 = (
            node_trace1[key].index[node_trace1[key]["count"] > frequency].values
        )
        trace_set2 = (
            node_trace2[key].index[node_trace2[key]["count"] > frequency].values
        )
        intersect = np.intersect1d(trace_set1, trace_set2)
        union = np.union1d(trace_set1, trace_set2)
        if len(union) != 0:
            return len(intersect) / len(union)
        else:
            return 0
    else:
        return None


def calc_iou_per_layer(
    trace1: AttrMap, trace2: AttrMap, node_name: str, key: str = TraceKey.EDGE
) -> float:
    node_trace1 = trace1.nodes[node_name]
    if key in node_trace1:
        node_trace2 = trace2.nodes[node_name]
        trace_set1 = TraceKey.to_array(node_trace1[key])
        trace_set2 = TraceKey.to_array(node_trace2[key])
        intersect = np.intersect1d(trace_set1, trace_set2)
        union = np.union1d(trace_set1, trace_set2)
        return len(intersect) / len(union)
    else:
        return None


def calc_class_trace_side_overlap(
    class_trace: AttrMap, trace: AttrMap, key: str = TraceKey.EDGE
) -> float:
    def intersect(node_name: str) -> Optional[int]:
        node_class_trace = class_trace.nodes[node_name]
        if key in node_class_trace:
            node_trace = trace.nodes[node_name]
            class_trace_set = TraceKey.to_array(node_class_trace[key])
            trace_set = TraceKey.to_array(node_trace[key])
            intersect = np.intersect1d(class_trace_set, trace_set)
            return len(intersect)
        else:
            return None

    iou = (
        sum(filter_not_null([intersect(node_name) for node_name in class_trace.nodes]))
        / class_trace.attrs[TraceKey.max_of(TraceKey.num_of(key))]
    )
    return iou


def calc_class_trace_side_overlap_norm(
    class_trace: AttrMap, trace: AttrMap, key: str = TraceKey.EDGE
) -> float:
    def intersect(node_name: str) -> Optional[int]:
        node_class_trace = class_trace.nodes[node_name]
        if key in node_class_trace:
            node_trace = trace.nodes[node_name]
            class_trace_set = TraceKey.to_array(node_class_trace[key])
            trace_set = TraceKey.to_array(node_trace[key])
            intersect = np.intersect1d(class_trace_set, trace_set)
            return len(intersect)
        else:
            return None

    iou = (
        sum(filter_not_null([intersect(node_name) for node_name in class_trace.nodes]))
        - class_trace.attrs[TraceKey.min_of(TraceKey.num_of(key))]
    ) / class_trace.attrs[TraceKey.max_of(TraceKey.num_of(key))]
    return iou


def calc_trace_side_overlap(
    class_trace: AttrMap,
    trace: AttrMap,
    key: str = TraceKey.EDGE,
    node_name: str = None,
) -> float:
    def intersect_and_union(node_name: str) -> Optional[Tuple[int, int]]:
        node_class_trace = class_trace.nodes[node_name]
        if key in node_class_trace:
            # if key == TraceKey.EDGE and node_name.startswith("max"):
            #     return None
            node_trace = trace.nodes[node_name]
            class_trace_set = TraceKey.to_array(node_class_trace[key])
            trace_set = TraceKey.to_array(node_trace[key])
            intersect = np.intersect1d(class_trace_set, trace_set)
            return len(intersect), len(trace_set)
        else:
            return None

    def get_iou(args: Tuple[int, int]) -> float:
        intersect_size, union_size = args
        return intersect_size / union_size

    if node_name is None:
        iou = get_iou(
            reduce(
                lambda x, y: (x[0] + y[0], x[1] + y[1]),
                filter_not_null(
                    [intersect_and_union(node_name) for node_name in class_trace.nodes]
                ),
            )
        )
    else:
        iou = intersect_and_union(node_name)
        if iou is not None:
            iou = get_iou(iou)
    return iou


def calc_trace_size(
    trace: AttrMap, key: str = TraceKey.EDGE, compact: bool = False
) -> Optional[int]:
    def trace_size(node_name: str) -> Optional[int]:
        node_trace = trace.nodes[node_name]
        if key in node_trace:
            if compact:
                return np.count_nonzero(np.unpackbits(node_trace[key]))
            else:
                return TraceKey.to_array(node_trace[key]).size
        else:
            return None

    return sum(filter_not_null([trace_size(node_name) for node_name in trace.nodes]))


def calc_trace_size_per_layer(
    trace: AttrMap, layer_name: str, key: str = TraceKey.EDGE, compact: bool = False
) -> Optional[int]:
    def trace_size(node_name: str) -> Optional[int]:
        node_trace = trace.nodes[node_name]
        if key in node_trace:
            if compact:
                return np.count_nonzero(np.unpackbits(node_trace[key]))
            else:
                return TraceKey.to_array(node_trace[key]).size
        else:
            return None

    return trace_size(layer_name)


def calc_trace_path_num(trace: AttrMap, layer: str) -> int:
    return trace.tensors[layer][TraceKey.PATH]["count"].sum()


def calc_trace_side_overlap_compact(
    class_trace: AttrMap,
    trace: AttrMap,
    key: str = TraceKey.EDGE,
    node_name: str = None,
) -> float:
    def intersect_and_union(node_name: str) -> Optional[Tuple[int, int]]:
        node_class_trace = class_trace.nodes[node_name]
        if key in node_class_trace:
            node_trace = trace.nodes[node_name]
            class_trace_set = TraceKey.to_array(
                np.argwhere(np.unpackbits(node_class_trace[key]))
            )
            trace_set = TraceKey.to_array(node_trace[key])
            intersect = np.intersect1d(class_trace_set, trace_set)
            return len(intersect), len(trace_set)
        else:
            return None

    def get_iou(args: Tuple[int, int]) -> float:
        intersect_size, union_size = args
        if union_size == 0:
            return 0
        else:
            return intersect_size / union_size

    if node_name is None:
        iou = get_iou(
            reduce(
                lambda x, y: (x[0] + y[0], x[1] + y[1]),
                filter_not_null(
                    [intersect_and_union(node_name) for node_name in class_trace.nodes]
                ),
            )
        )
    else:
        iou = intersect_and_union(node_name)
        if iou is not None:
            iou = get_iou(iou)
    return iou


def calc_trace_side_overlap_both_compact(
    class_trace: AttrMap,
    trace: AttrMap,
    key: str = TraceKey.EDGE,
    node_name: str = None,
    return_size: bool = False,
) -> float:
    def intersect_and_union(node_name: str) -> Optional[Tuple[int, int]]:
        node_class_trace = class_trace.nodes[node_name]
        if key in node_class_trace:
            node_trace = trace.nodes[node_name]
            class_trace_set = node_class_trace[key]
            trace_set = node_trace[key]
            intersect = np.bitwise_and(class_trace_set, trace_set)
            return (
                np.count_nonzero(np.unpackbits(intersect)),
                np.count_nonzero(np.unpackbits(trace_set)),
            )
        else:
            return None

    def get_iou(args: Tuple[int, int]) -> float:
        intersect_size, union_size = args
        if union_size == 0:
            return 0
        else:
            return intersect_size / union_size

    if node_name is None:
        intersect_size, union_size = reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1]),
            filter_not_null(
                [intersect_and_union(node_name) for node_name in class_trace.nodes]
            ),
        )
        iou = get_iou((intersect_size, union_size))
        if return_size:
            return iou, intersect_size
    else:
        iou = intersect_and_union(node_name)
        if iou is not None:
            iou = get_iou(iou)
    return iou


def calc_class_trace_side_overlap_compact(
    class_trace: AttrMap,
    trace: AttrMap,
    key: str = TraceKey.EDGE,
    node_name: str = None,
) -> float:
    def intersect_and_union(node_name: str) -> Optional[Tuple[int, int]]:
        node_class_trace = class_trace.nodes[node_name]
        if key in node_class_trace:
            node_trace = trace.nodes[node_name]
            class_trace_set = TraceKey.to_array(
                np.argwhere(np.unpackbits(node_class_trace[key]))
            )
            trace_set = TraceKey.to_array(node_trace[key])
            intersect = np.intersect1d(class_trace_set, trace_set)
            return len(intersect), len(class_trace_set)
        else:
            return None

    def get_iou(args: Tuple[int, int]) -> float:
        intersect_size, union_size = args
        if union_size == 0:
            return 0
        else:
            return intersect_size / union_size

    if node_name is None:
        iou = get_iou(
            reduce(
                lambda x, y: (x[0] + y[0], x[1] + y[1]),
                filter_not_null(
                    [intersect_and_union(node_name) for node_name in class_trace.nodes]
                ),
            )
        )
    else:
        iou = intersect_and_union(node_name)
        if iou is not None:
            iou = get_iou(iou)
    return iou


def calc_weighted_iou(
    class_trace: AttrMap,
    trace: AttrMap,
    key: str = TraceKey.EDGE,
    node_name: str = None,
) -> float:
    def intersect_and_union(node_name: str) -> Optional[Tuple[float, float]]:
        node_class_trace = class_trace.nodes[node_name]
        if key in node_class_trace:
            node_trace = trace.nodes[node_name]
            frame_class_trace = node_class_trace[key]
            intersect_mask = np.isin(
                TraceKey.to_array(frame_class_trace),
                node_trace[key],
                assume_unique=True,
            )
            return (
                frame_class_trace["count"].values[intersect_mask].sum()
                / class_trace.attrs[TraceKey.COUNT],
                frame_class_trace["count"].values.sum()
                / class_trace.attrs[TraceKey.COUNT],
            )
        else:
            return None

    def get_iou(args: Tuple[float, float]) -> float:
        intersect_size, union_size = args
        return intersect_size / union_size

    if node_name is None:
        iou = get_iou(
            reduce(
                lambda x, y: (x[0] + y[0], x[1] + y[1]),
                filter_not_null(
                    [intersect_and_union(node_name) for node_name in class_trace.nodes]
                ),
            )
        )
    else:
        iou = intersect_and_union(node_name)
        if iou is not None:
            iou = get_iou(iou)
    return iou


def calc_iou_compact(
    trace1: AttrMap, trace2: AttrMap, key: str = TraceKey.EDGE
) -> float:
    def intersect_and_union(node_name: str) -> Optional[Tuple[int, int]]:
        node_trace1 = trace1.nodes[node_name]
        if key in node_trace1:
            node_trace2 = trace2.nodes[node_name]
            trace_set1 = node_trace1[key]
            trace_set2 = node_trace2[key]
            intersect = np.bitwise_and(trace_set1, trace_set2)
            union = np.bitwise_or(trace_set1, trace_set2)
            return (
                np.count_nonzero(np.unpackbits(intersect)),
                np.count_nonzero(np.unpackbits(union)),
            )
        else:
            return None

    def get_iou(args: Tuple[int, int]) -> float:
        intersect_size, union_size = args
        if union_size == 0:
            return 0
        else:
            return intersect_size / union_size

    iou = get_iou(
        reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1]),
            filter_not_null(
                [intersect_and_union(node_name) for node_name in trace1.nodes]
            ),
        )
    )
    return iou


def calc_iou_compact_per_layer(
    trace1: AttrMap, trace2: AttrMap, node_name: str, key: str = TraceKey.EDGE
) -> float:
    node_trace1 = trace1.nodes[node_name]
    if key in node_trace1:
        node_trace2 = trace2.nodes[node_name]
        trace_set1 = node_trace1[key]
        trace_set2 = node_trace2[key]
        intersect = np.bitwise_and(trace_set1, trace_set2)
        union = np.bitwise_or(trace_set1, trace_set2)
        return np.count_nonzero(np.unpackbits(intersect)) / np.count_nonzero(
            np.unpackbits(union)
        )
    else:
        return None


def self_similarity_matrix(
    iterable: Iterable[T],
    trace_fn: Callable[[T], AttrMap],
    similarity_fn: Callable[[AttrMap, AttrMap], float],
) -> np.ndarray:
    if not isinstance(iterable, list):
        iterable = list(iterable)
    size = len(iterable)
    matrix = np.eye(size, dtype=float)
    for i in range(0, size):
        for j in range(i + 1, size):
            trace_i = trace_fn(iterable[i])
            trace_j = trace_fn(iterable[j])
            similarity = similarity_fn(trace_i, trace_j)
            matrix[i][j] = similarity
            matrix[j][i] = similarity
    return matrix


def self_similarity_matrix_ray(
    partial_path: str,
    iterable: Iterable[T],
    trace_fn: Callable[[T], AttrMap],
    similarity_fn: Callable[[AttrMap, AttrMap], float],
    key: str = TraceKey.EDGE,
) -> np.ndarray:
    if not isinstance(iterable, list):
        iterable = list(iterable)
    size = len(iterable)

    def calc_similarity(iter_i, iter_j):
        trace_i = trace_fn(iter_i)
        trace_j = trace_fn(iter_j)
        if trace_i is None or trace_j is None:
            return 0.0
        else:
            similarity = similarity_fn(trace_i, trace_j, key=key)
            return similarity

    def save_and_load_similarity(i, j):
        # tr.print_diff()
        iter_i = iterable[i]
        iter_j = iterable[j]
        action = IOAction(
            f"{partial_path}/{iter_i}_{iter_j}.pkl",
            init_fn=lambda: calc_similarity(iter_i, iter_j),
            cache=True,
        )
        action.save()
        return i, j, action.load()

    # tr = tracker.SummaryTracker()
    similarity_list = ray_iter(
        save_and_load_similarity,
        [(i, j) for i in range(0, size) for j in range(i + 1, size)],
        out_of_order=True,
        chunksize=1,
    )
    matrix = np.eye(size, dtype=float)
    for i, j, similarity in similarity_list:
        matrix[i][j] = similarity
        matrix[j][i] = similarity
        print(f"finish i={i}, j={j}")
    return matrix


def inter_class_similarity_matrix_ray(
    partial_path: str,
    iterable: Iterable[T],
    trace_fn: Callable[[T, str], AttrMap],
    similarity_fn: Callable[[AttrMap, AttrMap], float],
    key: str = TraceKey.EDGE,
) -> np.ndarray:
    if not isinstance(iterable, list):
        iterable = list(iterable)
    size = len(iterable)

    def calc_similarity(iter_i, iter_j):
        trace_i = trace_fn(iter_i, "left")
        trace_j = trace_fn(iter_j, "right")
        if trace_i is None or trace_j is None:
            return 0.0
        else:
            similarity = similarity_fn(trace_i, trace_j, key=key)
            del trace_i
            del trace_j
            gc.collect()
            return similarity

    def save_and_load_similarity(i, j):
        # tr.print_diff()
        iter_i = iterable[i]
        iter_j = iterable[j]
        action = IOAction(
            f"{partial_path}/{iter_i}_{iter_j}.pkl",
            init_fn=lambda: calc_similarity(iter_i, iter_j),
            cache=True,
        )
        action.save()
        return i, j, action.load()

    # tr = tracker.SummaryTracker()
    similarity_list = ray_iter(
        save_and_load_similarity,
        [(i, j) for i in range(0, size) for j in range(i, size)],
        out_of_order=True,
        chunksize=1,
    )
    matrix = np.zeros((size, size), dtype=float)
    for i, j, similarity in similarity_list:
        matrix[i][j] = similarity
        matrix[j][i] = similarity
        print(f"finish i={i}, j={j}")
    return matrix


def calc_density(trace: AttrMap, key: str) -> float:
    density = sum(
        node[key].size for name, node in trace.nodes.items() if key in node
    ) / sum(
        np.prod(node[key + "_shape"])
        for name, node in trace.nodes.items()
        if key in node
    )
    return density


def calc_density_compact(trace: AttrMap, key: str) -> float:
    density = sum(
        np.count_nonzero(np.unpackbits(node[key]))
        for name, node in trace.nodes.items()
        if key in node
    ) / sum(
        np.prod(node[key + "_shape"])
        for name, node in trace.nodes.items()
        if key in node
    )
    return density


def calc_density_compact_per_layer(
    trace: AttrMap, layers: List[str], key: str
) -> pd.DataFrame:
    result_layers = []
    densities = []
    for layer_name in layers:
        node = trace.nodes[layer_name]
        if key in node:
            result_layers.append(layer_name)
            densities.append(
                np.count_nonzero(np.unpackbits(node[key]))
                / np.prod(node[key + "_shape"])
            )
    return pd.DataFrame(dict(density=densities), index=result_layers).rename_axis(
        "layer"
    )


def calc_metrics_compact_per_layer(trace: AttrMap, layers: List[str]) -> pd.DataFrame:
    result_layers = []
    metrics = []
    for layer_name in layers:
        node = trace.nodes[layer_name]
        for metric_name in TraceKey.METRICS:
            result_layers.append(f"{layer_name}/{metric_name}")
            metrics.append(node[metric_name])
    return pd.DataFrame(dict(value=metrics), index=result_layers).rename_axis(
        "layer_metric"
    )


def calc_skip_ratio(graph: Graph, layers: List[str]) -> pd.DataFrame:
    result_layers = []
    skip_ratios = []
    for node_name in layers:
        node = graph.node(graph.id(node_name))
        if isinstance(node, AddOp):
            traced_edges = np.unpackbits(node.attrs[TraceKey.EDGE]).reshape(
                node.attrs[TraceKey.EDGE_SHAPE]
            )
            result_layers.append(node.name)
            skip_ratios.append(
                np.count_nonzero(traced_edges[1]) / np.count_nonzero(traced_edges)
            )
    return pd.DataFrame(dict(skip_ratio=skip_ratios), index=result_layers).rename_axis(
        "layer"
    )


def calc_space(trace: AttrMap, key: str) -> int:
    return sum(
        np.prod(node[key + "_shape"])
        for name, node in trace.nodes.items()
        if key in node
    )
