import itertools
from functools import partial

import pandas as pd

from nninst.backend.tensorflow.attack.common import imagenet_example_trace
from nninst.backend.tensorflow.model.config import (
    ALEXNET,
    LENET,
    RESNET_18_CIFAR10,
    RESNET_18_CIFAR100,
    RESNET_50,
    ModelConfig,
)
from nninst.op import Conv2dOp, DenseOp
from nninst.statistics import (
    calc_density_compact_per_layer,
    calc_metrics_compact_per_layer,
)
from nninst.trace import TraceKey, get_trace, get_type4_trace
from nninst.utils import filter_not_null
from nninst.utils.fs import CsvIOAction


def trace_per_layer_metrics(
    model_config: ModelConfig,
    threshold: float,
    train: bool = True,
    absolute_threshold: float = None,
) -> CsvIOAction:
    def get_trace_per_layer_metrics():
        per_layer_metrics = lambda: get_per_layer_metrics(
            model_config, threshold=threshold, absolute_threshold=absolute_threshold
        )
        traces = filter_not_null(
            imagenet_example_trace(
                model_config=model_config,
                attack_name="original",
                attack_fn=None,
                generate_adversarial_fn=None,
                trace_fn=get_trace
                if absolute_threshold is None
                else partial(
                    get_type4_trace,
                    output_threshold=per_layer_metrics(),
                    input_threshold=per_layer_metrics(),
                ),
                class_id=class_id,
                image_id=image_id,
                threshold=threshold,
                train=train,
            ).load()
            for class_id in model_config.class_list()
            # for class_id in range(4)
            for image_id in range(model_config.image_num_per_class)
        )
        graph = model_config.network_class.graph().load()
        layers = graph.ops_in_layers(DenseOp, Conv2dOp)
        metrics_of_traces = list(
            map(
                lambda trace: calc_metrics_compact_per_layer(trace, layers=layers),
                traces,
            )
        )
        df = pd.concat(metrics_of_traces)
        return df.groupby(df.index).mean()
        # .density.to_dict()

    path = f"metrics/{model_config.name}_trace_per_layer_metrics_{threshold:.1f}"
    if absolute_threshold:
        path = f"{path}_absolute_{absolute_threshold:.2f}"
    if train:
        path = f"{path}_train"
    path = f"{path}.csv"
    return CsvIOAction(path, init_fn=get_trace_per_layer_metrics, cache=True)


def get_per_layer_metrics(
    model_config: ModelConfig, threshold: float, absolute_threshold: float = None
):
    metrics = (
        trace_per_layer_metrics(model_config, threshold, train=True)
        .load()
        .set_index("layer_metric")
        .value.to_dict()
    )
    if absolute_threshold is not None:
        for key, value in metrics.items():
            if key.endswith(TraceKey.RECEPTIVE_FIELD_THRESHOLD):
                metrics[key] = absolute_threshold
    metrics["threshold"] = threshold
    return metrics


def calc_per_layer_metrics(model_config: ModelConfig):
    for threshold, absolute_threshold in itertools.product(
        [
            # 1.0,
            # 0.9,
            # 0.7,
            0.5,
            # 0.3,
            # 0.1,
        ],
        [
            None,
            # 0.05,
            # 0.1,
            # 0.2,
            # 0.3,
            # 0.4,
        ],
    ):
        density = trace_per_layer_metrics(
            model_config, threshold, absolute_threshold=absolute_threshold
        )
        density.save()
        print(density.load())


if __name__ == "__main__":
    model_config = ALEXNET
    # model_config = RESNET_18_CIFAR100
    # model_config = RESNET_18_CIFAR10
    # model_config = LENET
    # model_config = RESNET_50
    calc_per_layer_metrics(model_config)
