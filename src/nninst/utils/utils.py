import itertools
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, Optional, Tuple, TypeVar

import psutil

__all__ = [
    "join_not_null",
    "filter_not_null",
    "filter_value_not_null",
    "grouper",
    "pool",
    "merge_dict",
    "to_suffix",
    "assert_runnable",
    "map_prefix",
]


def join_not_null(iterable: Iterable[str], split: str = ", ") -> str:
    return split.join([element for element in iterable if element is not None])


T = TypeVar("T")


def filter_not_null(iterable: Iterable[T]) -> Iterable[T]:
    return (element for element in iterable if element is not None)


def grouper(n: int, iterable: Iterable[T]) -> Iterable[Tuple[T, ...]]:
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


K = TypeVar("K")
V = TypeVar("V")


def filter_value_not_null(map: Dict[K, Optional[V]]) -> Dict[K, V]:
    return {key: value for key, value in map.items() if value is not None}


def merge_dict(*dicts: Dict[K, V]) -> Dict[K, V]:
    if len(dicts) == 0:
        return {}
    elif len(dicts) == 1:
        return dicts[0]
    else:
        merged_dict = {}
        for map in dicts:
            merged_dict.update(map)
        return merged_dict


# @functools.lru_cache()
def pool(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers)


def to_suffix(name: str) -> str:
    # if name == "" or name is None:
    if name == "":
        return ""
    else:
        return f"_{name}"


def assert_memory_available():
    if psutil.virtual_memory().percent > 95:
        raise RuntimeError(
            f"available memory is less than 10% in {socket.gethostname()}"
        )


def assert_not_stop():
    path = os.path.abspath("stop")
    if os.path.exists(path):
        raise RuntimeError("stop file exists")


def assert_runnable():
    assert_memory_available()
    assert_not_stop()


def map_prefix(map: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}.{key}": value for key, value in map.items()}
