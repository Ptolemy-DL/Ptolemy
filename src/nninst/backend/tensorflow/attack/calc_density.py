import pandas as pd

from nninst.backend.tensorflow.attack.common import imagenet_example_trace
from nninst.backend.tensorflow.model.config import ALEXNET, RESNET_50, ModelConfig
from nninst.op import Conv2dOp, DenseOp
from nninst.statistics import calc_density_compact_per_layer
from nninst.trace import TraceKey, get_trace
from nninst.utils import filter_not_null
from nninst.utils.fs import CsvIOAction


def trace_density(
    model_config: ModelConfig, threshold: float, key: str, train: bool = True
) -> CsvIOAction:
    def get_trace_density():
        traces = filter_not_null(
            imagenet_example_trace(
                model_config=model_config,
                attack_name="original",
                attack_fn=None,
                generate_adversarial_fn=None,
                trace_fn=get_trace,
                class_id=class_id,
                image_id=image_id,
                threshold=threshold,
                train=train,
            ).load()
            for class_id in model_config.class_list()
            # for class_id in range(4)
            for image_id in range(1)
        )
        graph = model_config.network_class.graph().load()
        layers = graph.ops_in_layers(DenseOp, Conv2dOp)
        if key == TraceKey.POINT:
            layers = list(
                map(
                    lambda layer: graph.op(graph.id(layer)).output_nodes[0].name, layers
                )
            )
        density_of_traces = list(
            map(
                lambda trace: calc_density_compact_per_layer(
                    trace, layers=layers, key=key
                ),
                traces,
            )
        )
        df = pd.concat(density_of_traces)
        return df.groupby(df.index).mean()
        # .density.to_dict()

    path = f"metrics/imagenet_{model_config.name}_trace_density_{threshold:.1f}_{key}"
    if train:
        path = f"{path}_train"
    path = f"{path}.csv"
    return CsvIOAction(path, init_fn=get_trace_density, cache=True)


if __name__ == "__main__":
    model_config = ALEXNET
    # model_config = RESNET_50
    threshold = 0.5
    key = TraceKey.POINT
    density = trace_density(model_config, threshold, key)
    density.save()
    print(density.load())
