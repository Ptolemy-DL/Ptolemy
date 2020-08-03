#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

import os
from functools import partial, reduce

import tensorflow as tf

from nninst import AttrMap, mode
from nninst.backend.tensorflow.dataset import imagenet
from nninst.backend.tensorflow.graph import model_fn_with_fetch_hook
from nninst.backend.tensorflow.model import ResNet50
from nninst.backend.tensorflow.trace.common import (
    reconstruct_class_trace_from_tf,
    self_similarity,
)
from nninst.statistics import calc_density
from nninst.trace import TraceKey, merge_compact_trace, merge_trace
from nninst.utils import filter_not_null, grouper
from nninst.utils.fs import IOAction, abspath
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init, ray_iter, ray_map, ray_map_reduce

__all__ = ["resnet_50_imagenet_class_trace", "resnet_50_imagenet_self_similarity"]


def resnet_50_imagenet_class_trace(
    class_id: int, threshold: float, label: str = None, compress: bool = False
) -> IOAction[AttrMap]:
    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "resnet_50_imagenet"
    else:
        name = f"resnet_50_imagenet_{label}"
    path = f"store/analysis/class_trace/{name}/approx_{threshold_name}/{class_id}.pkl"
    return IOAction(path, init_fn=None, compress=compress)


def save_all_resnet_50_imagenet_class_traces(threshold: float, label: str = None):
    def get_trace(class_id: int, batch_id: int, batch_size: int = 1) -> AttrMap:
        try:
            mode.check(False)
            data_dir = abspath("/home/yxqiu/data/imagenet")
            trace = reconstruct_class_trace_from_tf(
                model_fn=partial(
                    model_fn_with_fetch_hook,
                    create_model=lambda: ResNet50(),
                    graph=ResNet50.graph().load(),
                ),
                input_fn=lambda: imagenet.train(
                    data_dir,
                    batch_size,
                    transform_fn=lambda dataset: dataset.filter(
                        lambda image, label: tf.equal(
                            tf.convert_to_tensor(class_id, dtype=tf.int32), label
                        )
                    )
                    .skip(batch_id * batch_size)
                    .take(batch_size),
                ),
                model_dir=abspath("tf/resnet-50-v2/model/"),
                select_fn=lambda input: arg_approx(input, threshold),
                class_id=class_id,
                parallel=4,
            )
            # trace = AttrMap()
            return trace
        except Exception as cause:
            raise RuntimeError(
                f"error when handling class {class_id} batch {batch_id}"
            ) from cause

    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "resnet_50_imagenet"
    else:
        name = f"resnet_50_imagenet_{label}"
    prefix = f"store/analysis/class_trace/{name}/approx_{threshold_name}"
    class_ids = [
        class_id
        for class_id in range(1, 1001)
        if not os.path.exists(f"{prefix}/{class_id}.pkl")
    ]
    image_num = 140
    batch_size = 1
    batch_num = image_num // batch_size
    traces = ray_map_reduce(
        get_trace,
        merge_trace,
        [
            (
                class_id,
                [(class_id, batch_id, batch_size) for batch_id in range(0, batch_num)],
            )
            for class_id in class_ids
        ],
        num_gpus=0,
    )
    for class_id, trace in traces:
        IOAction(f"{prefix}/{class_id}.pkl", init_fn=lambda: trace).save()
        print(f"finish class {class_id}")
    # merged_traces = defaultdict(lambda: None)
    # trace_count = defaultdict(int)
    # for next_trace in traces:
    #     class_id, batch_id, trace = next_trace
    #     print(f"begin class {class_id} batch {batch_id}")
    #     if trace is not None:
    #         merged_traces[class_id] = merge_trace(trace, merged_traces[class_id])
    #     # print(f"finish class {class_id} image {batch_id}")
    #     trace_count[class_id] = trace_count[class_id] + 1
    #     if trace_count[class_id] == batch_num:
    #         IOAction(f"{prefix}/{class_id}.pkl", init_fn=lambda: merged_traces[class_id]).save()
    #         del trace_count[class_id]
    #         print(f"============== finish class {class_id}")


def resnet_50_imagenet_trace(threshold: float, label: str = None) -> IOAction[AttrMap]:
    def get_trace() -> AttrMap:
        def get_class_trace(class_id):
            try:
                return resnet_50_imagenet_class_trace(
                    class_id,
                    threshold=threshold,
                    label="compact" if label is None else label + "_compact",
                    compress=True,
                ).load()
            except Exception as cause:
                raise RuntimeError(f"raise from class {class_id}") from cause

        # return list(ray_map_reduce(get_class_trace, merge_compact_trace,
        #                            [["all", range(1, 1001)]]))[0][1]
        def merge(class_ids):
            return reduce(
                merge_compact_trace,
                (get_class_trace(class_id) for class_id in class_ids),
            )

        return reduce(
            merge_compact_trace,
            ray_iter(
                merge, grouper(50, range(1, 1001)), out_of_order=True, huge_task=True
            ),
        )

    threshold_name = "{0:.3f}".format(threshold)
    if label is None:
        name = "resnet_50_imagenet"
    else:
        name = f"resnet_50_imagenet_{label}"
    path = f"store/analysis/trace/{name}/approx_{threshold_name}/trace.pkl"
    return IOAction(path, init_fn=get_trace, compress=True)


def check_class_trace(
    threshold: float, label: str = None, compress: bool = False, start: int = 1
):
    def check(class_id):
        action = resnet_50_imagenet_class_trace(
            class_id, threshold=threshold, label=label, compress=compress
        )
        if action.is_saved():
            try:
                action.load()
            except EOFError:
                os.remove(action.path)
                return class_id
            except Exception as cause:
                raise RuntimeError(f"raise in class {class_id}") from cause

    corrupt_traces = filter_not_null(
        ray_map(check, range(start, 1001), out_of_order=True)
    )
    print(f"corrupt traces: {corrupt_traces}")


# def resnet_50_imagenet_self_similarity(
#     threshold: float, label: str = None
# ) -> IOAction[np.ndarray]:
#     def get_self_similarity() -> np.ndarray:
#         return self_similarity_matrix_ray(
#             partial_path,
#             range(1, 1001, 10),
#             trace_fn=lambda class_id: resnet_50_imagenet_class_trace(
#                 class_id, threshold, "compact" if label is None else label + "_compact", compress=True
#             ).load(),
#             similarity_fn=calc_iou_compact
#         )
#
#     threshold_name = "{0:.3f}".format(threshold)
#     if label is None:
#         name = "resnet_50_imagenet"
#     else:
#         name = f"resnet_50_imagenet_{label}"
#     prefix = f"store/analysis/self_similarity/{name}/approx_{threshold_name}/"
#     path = f"{prefix}/self_similarity.pkl"
#     partial_path = f"{prefix}/partial/"
#     return IOAction(path, init_fn=get_self_similarity)


resnet_50_imagenet_self_similarity = self_similarity(
    name="resnet_50_imagenet",
    trace_fn=resnet_50_imagenet_class_trace,
    class_ids=range(1, 1001, 10),
)

if __name__ == "__main__":
    # mode.check(False)
    # mode.debug()
    mode.local()
    # mode.distributed()
    ray_init("gpu")
    threshold = 0.5
    # threshold = 1
    # threshold = 0.8

    # label = None
    label = "train_50"
    # label = "train_start"
    # label = "train_start_more"

    print(f"generate class trace for label {label}")

    # save_all_resnet_50_imagenet_class_traces(threshold, label)

    # resnet_50_imagenet_trace(threshold, label).save()

    # check_class_trace(threshold, "compact", compress=True)
    # check_class_trace(threshold, "train_50_compact", compress=True)
    # check_class_trace(threshold, "train_50_compress", compress=True)

    trace = resnet_50_imagenet_class_trace(1, threshold, label).load()
    # trace = resnet_50_imagenet_class_trace(1, threshold, label, compress=True).load()
    for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
        print(f"{key}: {calc_density(trace, key)}")
    # # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    # #     print(f"{key}: {calc_space(trace, key)}")

    # trace = resnet_50_imagenet_trace(threshold, label).load()
    # for key in [TraceKey.POINT, TraceKey.EDGE, TraceKey.WEIGHT]:
    #     print(f"{key}: {calc_density_compact(trace, key)}")

    # trace = compact_trace(trace, ResNet50.graph().load())
    # start_time = time.time()
    # dumped_trace = pickle.dumps(trace)
    # after_dump = time.time()
    # print(f"dump: {len(dumped_trace)}, time: {after_dump - start_time}")
    # compressed_trace = zlib.compress(dumped_trace, level=9)
    # # compressed_trace = lzma.compress(dumped_trace)
    # after_compress = time.time()
    # print(f"dump+compress: {len(compressed_trace)}, time: {after_compress - after_dump}")
    # decompressed_trace = zlib.decompress(compressed_trace)
    # # decompressed_trace = lzma.decompress(compressed_trace)
    # after_decompress = time.time()
    # print(f"decompress time: {after_decompress - after_compress}")

    # resnet_50_imagenet_self_similarity(threshold, label).save()
    # similarity_matrix = resnet_50_imagenet_self_similarity(threshold, label).load()
