import argparse
import sys
from functools import partial

from nninst.trace import get_per_input_unstructured_trace, get_trace, get_type2_trace, get_type4_trace


def parse_path_generation_args(model_config):
    from nninst.backend.tensorflow.attack.calc_per_layer_metrics import (
        get_per_layer_metrics,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="EP",
        help="Different types of path extraction, default EP, pick between BwCU, BwAB and FwAB",
    )
    parser.add_argument(
        "--cumulative_threshold",
        type=float,
        default=0.5,
        help="cumulative threshold theta, default 0.5",
    )
    parser.add_argument(
        "--absolute_threshold",
        type=float,
        default=0,
        help="absolute threshold phi, default None",
    )
    params, unparsed = parser.parse_known_args()
    if params.absolute_threshold == 0:
        absolute_threshold = None
    else:
        absolute_threshold = params.absolute_threshold
    cumulative_threshold = params.cumulative_threshold
    per_layer_metrics = lambda: get_per_layer_metrics(
        model_config, threshold=cumulative_threshold, absolute_threshold=absolute_threshold
    )
    if params.type == "EP":
        type_ = [get_trace, None, None, None]
    elif params.type == "BwCU":
        type_ = [
            partial(get_type2_trace, output_threshold=per_layer_metrics()),
            f"type2_density_from_{cumulative_threshold:.1f}",
            "type2_trace",
            f"density_from_{cumulative_threshold:.1f}",
        ]
    elif params.type == "BwAB":
        type_ = [
            partial(
                get_type4_trace,
                output_threshold=per_layer_metrics(),
                input_threshold=per_layer_metrics(),
            ),
            f"type4_density_from_{cumulative_threshold:.1f}_absolute_{absolute_threshold:.2f}",
            "type4_trace",
            f"density_from_{cumulative_threshold:.1f}_absolute_{absolute_threshold:.2f}",
        ]
    elif params.type == "FwAB":
        type_ = [
            partial(
                get_per_input_unstructured_trace,
                output_threshold=per_layer_metrics(),
                input_threshold=per_layer_metrics(),
            ),
            f"per_input_unstructured_density_from_{cumulative_threshold:.1f}",
            "per_input_unstructured_class_trace",
            f"density_from_{cumulative_threshold:.1f}",
        ]
    else:
        print("path construction type not supported")
        sys.exit()
    return absolute_threshold, cumulative_threshold, type_
