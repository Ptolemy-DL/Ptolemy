import os
import queue
import uuid
from collections import deque
from functools import wraps
from typing import TypeVar

import ray

from nninst import mode
from nninst.utils import assert_runnable, grouper
from nninst.utils.context import context

__all__ = [
    "lazy",
    "Lazy",
    "ray_get",
    "remote",
    "ray_iter",
    "ray_map",
    "ray_map_reduce",
    "ray_init",
]

_tag_to_ray = {
    "r730": 6370,
    "dell": 6371,
    "gpu": 6372,
    "gpu_only": 6373,
    # "tmp": 6379,
}


def ray_init(
    tag: str = None, num_cpus: int = None, num_gpus: int = None, **kwargs
) -> None:
    kwargs = dict(log_to_driver=False, **kwargs)
    if mode.is_debug():
        ray.init(local_mode=True, **kwargs)
    elif mode.is_local():
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_AFFINITY"] = "none"
        os.environ["KMP_SETTINGS"] = "0"
        ray.init(num_gpus=num_gpus, num_cpus=num_cpus, **kwargs)
    else:
        if tag is None:
            ray.init("dell-01:6378", **kwargs)
        else:
            ray.init(f"dell-01:{_tag_to_ray[tag]}", **kwargs)


def invoke_with_args(func, args):
    if isinstance(args, tuple):
        return func(*args)
    else:
        return func(args)


def chunked(func):
    def chunked_func(chunk):
        try:
            return [func(*args) for args in chunk]
        except TypeError:
            return [func(args) for args in chunk]

    return chunked_func


def disable_cache(func):
    def wrapper(ray_random_id, *args):
        if not isinstance(ray_random_id, uuid.UUID):
            raise RuntimeError(
                f"ray_random_id is {ray_random_id}, type: {type(ray_random_id)}, args: {args}"
            )
        return func(*args)

    return wrapper


def print_args(func):
    @wraps(func)
    def wrapper(*args):
        try:
            return func(*args)
        except Exception as cause:
            raise RuntimeError(
                f"exception occurs when running {func.__name__} with args {args}"
            ) from cause

    return wrapper


def ensure_runnable(func):
    @wraps(func)
    def wrapper(*args):
        assert_runnable()
        return func(*args)

    return wrapper


def switch_context(func):
    local_stack = context().configs.stack

    @wraps(func)
    def wrapper(*args):
        # if context().configs.lock.locked():
        #     raise RuntimeError(f"locked")
        remote_stack = context().configs.swap_stack(local_stack)
        try:
            # with context().configs.lock:
            #     return func(*args)
            return func(*args)
        finally:
            context().configs.swap_stack(remote_stack)

    return wrapper


# deprecated since Ray support spillback scheduling
def ray_iter_v1(
    func,
    iterable,
    out_of_order=False,
    chunksize=1,
    num_cpus=1,
    num_gpus=0,
    huge_task=False,
    pass_actor=False,
):
    func = print_args(func)
    func = ensure_runnable(func)
    func = switch_context(func)
    if chunksize == 1:
        if pass_actor:
            remote_func = remote(func, num_cpus, num_gpus)
        else:
            remote_func = remote(
                lambda args: invoke_with_args(func, args), num_cpus, num_gpus
            )
        iterator = iter(iterable)
    else:
        remote_func = remote(chunked(func), num_cpus, num_gpus)
        iterator = grouper(chunksize, iter(iterable))
    if mode.is_debug():
        device_num = 1
        parallel_num = 1
    else:
        clients = ray.nodes()
        if num_gpus != 0:
            device_num = int(
                sum(client["Resources"]["GPU"] for client in clients) / num_gpus
            )
        else:
            device_num = int(
                sum(client["Resources"]["CPU"] for client in clients) / num_cpus
            )
        parallel_num = device_num * 2
    if out_of_order:
        wait_num = 1 if huge_task else (device_num // 2 + 1)
        task_queue = []
        enqueue_num = 0
        dequeue_num = 0
        stop_iteration = False
        while True:
            if stop_iteration and enqueue_num == dequeue_num:
                break
            else:
                ready_ids, task_queue = ray.wait(
                    task_queue, num_returns=min(wait_num, len(task_queue))
                )
                while (not stop_iteration) and len(task_queue) < parallel_num:
                    try:
                        if pass_actor:
                            task_queue.append(remote_func(*next(iterator)))
                        else:
                            task_queue.append(remote_func(next(iterator)))
                        enqueue_num += 1
                    except StopIteration:
                        stop_iteration = True
                for ready_id in ready_ids:
                    if chunksize == 1:
                        yield ray_get(ready_id)
                    else:
                        for result in ray_get(ready_id):
                            yield result
                dequeue_num += len(ready_ids)
    else:
        task_queue = queue.Queue(parallel_num)
        enqueue_num = 0
        dequeue_num = 0
        stop_iteration = False
        while True:
            if stop_iteration and enqueue_num == dequeue_num:
                break
            else:
                while (not stop_iteration) and (not task_queue.full()):
                    try:
                        if pass_actor:
                            task_queue.put(remote_func(*next(iterator)))
                        else:
                            task_queue.put(remote_func(next(iterator)))
                        enqueue_num += 1
                    except StopIteration:
                        stop_iteration = True
                if task_queue.empty():
                    continue
                else:
                    if chunksize == 1:
                        yield ray_get(task_queue.get())
                    else:
                        for result in ray_get(task_queue.get()):
                            yield result
                    task_queue.task_done()
                    dequeue_num += 1


def ray_futures(func, iterable, chunksize=1, num_cpus=1, num_gpus=0, pass_actor=False):
    func = print_args(func)
    func = ensure_runnable(func)
    func = switch_context(func)
    if chunksize == 1:
        if pass_actor:
            remote_func = remote(func, num_cpus, num_gpus)
        else:
            remote_func = remote(
                lambda args: invoke_with_args(func, args), num_cpus, num_gpus
            )
        iterator = iter(iterable)
    else:
        remote_func = remote(chunked(func), num_cpus, num_gpus)
        iterator = grouper(chunksize, iter(iterable))
    return [
        remote_func(*task) if pass_actor else remote_func(task) for task in iterator
    ]


def ray_iter(
    func,
    iterable,
    out_of_order=False,
    chunksize=1,
    num_cpus=1,
    num_gpus=0,
    huge_task=False,
    pass_actor=False,
):
    remaining_ids = ray_futures(
        func=func,
        iterable=iterable,
        chunksize=chunksize,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        pass_actor=pass_actor,
    )
    if out_of_order:
        total_num = len(remaining_ids)
        finished_num = 0
        while len(remaining_ids) != 0:
            ready_ids, remaining_ids = ray.wait(remaining_ids)
            for object_id in ready_ids:
                finished_num += 1
                if chunksize == 1:
                    yield ray_get(object_id)
                else:
                    for result in ray_get(object_id):
                        yield result
        assert total_num == finished_num
    else:
        for object_id in remaining_ids:
            if chunksize == 1:
                yield ray_get(object_id)
            else:
                for result in ray_get(object_id):
                    yield result


def ray_map(
    func,
    iterable,
    out_of_order=False,
    chunksize=1,
    num_cpus=1,
    num_gpus=0,
    huge_task=False,
    pass_actor=False,
):
    return list(
        ray_iter(
            func,
            iterable,
            out_of_order,
            chunksize,
            num_cpus,
            num_gpus,
            huge_task,
            pass_actor,
        )
    )


def ray_reduce(func, iterable, num_cpus=1, num_gpus=0):
    func = print_args(func)
    func = ensure_runnable(func)
    func = switch_context(func)
    remote_reduce_func = remote(func, num_cpus, num_gpus)
    reduce_queue = deque(iterable)
    assert len(reduce_queue) > 0
    while len(reduce_queue) > 1:
        left_input = reduce_queue.popleft()
        right_input = reduce_queue.popleft()
        reduce_queue.append(remote_reduce_func(left_input, right_input))
    return ray_get(reduce_queue.popleft())


def ray_map_reduce(
    map_func, reduce_func, iterable, num_cpus=1, num_gpus=0, reduce_type: str = "tree"
):
    remote_map_func = remote(
        lambda args: invoke_with_args(map_func, args), num_cpus, num_gpus
    )
    remote_reduce_func = remote(reduce_func, num_cpus, num_gpus)
    iterator = iter(iterable)
    if mode.is_debug():
        device_num = 1
        parallel_num = 1
    else:
        schedulers = ray.global_state.local_schedulers()
        if num_gpus != 0:
            device_num = int(
                sum(scheduler["GPU"] for scheduler in schedulers) / num_gpus
            )
        else:
            device_num = int(
                sum(scheduler["CPU"] for scheduler in schedulers) / num_cpus
            )
        parallel_num = device_num * 2

    task_queue = []
    running_map_task_num = 0
    reduce_task_id_to_window_size = {}
    reduce_task_id_to_key = {}
    stop_iteration = False
    while True:
        if stop_iteration and running_map_task_num == 0:
            break
        else:
            # ready_ids, task_queue = ray.wait(task_queue, num_returns=(len(task_queue) // 4 + 1))
            ready_ids, task_queue = ray.wait(task_queue, num_returns=1)
            while (not stop_iteration) and running_map_task_num < parallel_num:
                try:
                    key, window = next(iterator)
                    reduce_list = []
                    for element in window:
                        reduce_list.append(remote_map_func(element))
                    window_size = len(reduce_list)
                    if reduce_type == "tree" and len(reduce_list) > 2:
                        reduce_queue = deque(reduce_list)
                        while len(reduce_queue) > 1:
                            left_input = reduce_queue.popleft()
                            right_input = reduce_queue.popleft()
                            reduce_queue.append(
                                remote_reduce_func(left_input, right_input)
                            )
                        reduce_task_id = reduce_queue.popleft()
                    elif reduce_type == "linear" and len(reduce_list) > 2:
                        reduce_queue = deque(reduce_list)
                        left_input = reduce_queue.popleft()
                        while len(reduce_queue) > 0:
                            right_input = reduce_queue.popleft()
                            left_input = remote_reduce_func(left_input, right_input)
                        reduce_task_id = left_input
                    else:
                        reduce_task_id = remote_reduce_func(*reduce_list)
                    task_queue.append(reduce_task_id)
                    reduce_task_id_to_window_size[reduce_task_id] = window_size
                    reduce_task_id_to_key[reduce_task_id] = key
                    running_map_task_num += window_size
                except StopIteration:
                    stop_iteration = True
            for ready_id in ready_ids:
                running_map_task_num -= reduce_task_id_to_window_size[ready_id]
                yield reduce_task_id_to_key[ready_id], ray_get(ready_id)


class Lazy:
    def __init__(self, object_id: ray.ObjectID) -> None:
        self.id = object_id


def lazy(object_id: ray.ObjectID) -> Lazy:
    return Lazy(object_id)


def ray_get(object_id):
    while True:
        value = ray.get(object_id)
        if isinstance(value, Lazy):
            object_id = value.id
        else:
            return value


T = TypeVar("T")


def remote_without_cache(func: T, num_cpus=1, num_gpus=0) -> T:
    func.__name__ = f"{func.__name__}_{uuid.uuid4()}"
    print(func.__name__)
    return ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(disable_cache(func)).remote


# @functools.lru_cache(maxsize=None)
def remote(func: T, num_cpus=1, num_gpus=0) -> T:
    func.__name__ = f"{func.__name__}_{uuid.uuid4()}"
    print(func.__name__)
    return ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(func).remote
