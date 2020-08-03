import time

from nninst.backend.tensorflow.trace.alexnet_imagenet_class_trace import (
    alexnet_imagenet_class_trace_compact,
)

if __name__ == "__main__":
    trace_type = None
    trace_parameter = None
    threshold = 0.5
    label = "import"
    variant = None
    total_time = 0
    for class_id in range(1, 11):
        start_time = time.time()
        alexnet_imagenet_class_trace_compact(
            class_id=class_id,
            trace_type=trace_type,
            trace_parameter=trace_parameter,
            threshold=threshold,
            label=label,
            variant=variant,
        ).load()
        used_time = time.time() - start_time
        total_time += used_time
    print(f"use: {total_time/10}s")
