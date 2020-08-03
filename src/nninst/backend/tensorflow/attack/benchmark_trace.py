import time
import timeit
from functools import partial

from nninst import mode
from nninst.backend.tensorflow.attack.calc_per_layer_metrics import (
    get_per_layer_metrics,
)
from nninst.backend.tensorflow.dataset import imagenet_raw
from nninst.backend.tensorflow.graph import model_fn_with_fetch_hook
from nninst.backend.tensorflow.model.config import ALEXNET
from nninst.backend.tensorflow.trace.common import reconstruct_trace_from_tf_v2
from nninst.dataset.envs import IMAGENET_RAW_DIR
from nninst.op import Conv2dOp
from nninst.trace import (
    get_per_input_unstructured_trace,
    get_per_receptive_field_unstructured_trace,
    get_trace,
    get_type2_trace,
    get_type3_trace,
    get_type4_trace,
    get_type7_trace,
    get_unstructured_trace,
)
from nninst.utils.fs import abspath
from nninst.utils.numpy import arg_approx, arg_sorted_topk


class MyTimer:
    def __init__(self):
        self.start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start
        msg = "The block took {time} seconds to complete"
        print(msg.format(time=runtime))


def benchmark_trace():
    rank = 1
    class_id = 1
    image_id = 0
    threshold = 0.5
    per_channel = False
    model_config = ALEXNET.with_model_dir("tf/alexnet/model_import")
    # model_config = RESNET_50
    # model_config = VGG_16
    mode.check(False)
    data_dir = IMAGENET_RAW_DIR
    model_dir = abspath(model_config.model_dir)
    create_model = lambda: model_config.network_class()
    graph = model_config.network_class.graph().load()
    model_fn = partial(model_fn_with_fetch_hook, create_model=create_model, graph=graph)
    # predicted_label = predict(
    #     create_model=create_model,
    #     input_fn=input_fn,
    #     model_dir=model_dir,
    # )
    #
    # if predicted_label != class_id:
    #     return None

    conv_op_count = 0

    def stop_hook(op):
        nonlocal conv_op_count
        if isinstance(op, Conv2dOp):
            conv_op_count += 1
        if conv_op_count >= 2:
            return True
        else:
            return False

    # reconstruct_trace_from_tf(
    #     class_id=class_id,
    #     model_fn=model_fn,
    #     input_fn=input_fn,
    #     select_fn=lambda input: arg_approx(input, threshold),
    #     model_dir=model_dir,
    #     per_channel=per_channel,
    #     # stop_hook=stop_hook,
    # )

    per_layer_metrics = lambda: get_per_layer_metrics(ALEXNET, threshold=0.5)
    for trace_fn in [
        get_trace,
        partial(get_type2_trace, output_threshold=per_layer_metrics()),
        partial(get_type3_trace, input_threshold=per_layer_metrics()),
        partial(
            get_type4_trace,
            output_threshold=per_layer_metrics(),
            input_threshold=per_layer_metrics(),
        ),
        partial(get_unstructured_trace, density=per_layer_metrics()),
        partial(
            get_per_receptive_field_unstructured_trace,
            output_threshold=per_layer_metrics(),
        ),
        partial(
            get_type7_trace,
            density=per_layer_metrics(),
            input_threshold=per_layer_metrics(),
        ),
        partial(
            get_per_input_unstructured_trace,
            output_threshold=per_layer_metrics(),
            input_threshold=per_layer_metrics(),
        ),
    ]:
        with MyTimer():
            for class_id in range(1, 11):
                input_fn = lambda: imagenet_raw.test(
                    data_dir,
                    class_id,
                    image_id,
                    class_from_zero=model_config.class_from_zero,
                    preprocessing_fn=model_config.preprocessing_fn,
                )
                reconstruct_trace_from_tf_v2(
                    model_fn=model_fn,
                    input_fn=input_fn,
                    trace_fn=partial(
                        trace_fn,
                        select_fn=lambda input: arg_approx(input, threshold),
                        select_seed_fn=lambda output: arg_sorted_topk(output, rank)[
                            rank - 1 : rank
                        ],
                    ),
                    model_dir=model_dir,
                    class_id=class_id,
                    rank=rank,
                )


if __name__ == "__main__":
    benchmark_trace()
