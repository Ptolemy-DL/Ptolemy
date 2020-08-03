from functools import partial
from typing import Any, Callable, Dict, Optional

import foolbox
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from foolbox.attacks import (
    FGSM,
    Attack,
    DeepFoolAttack,
    IterativeGradientSignAttack,
    SaliencyMapAttack,
)
from lenet_analysis import LenetAnalysis, get_class_per_layer_traced_edges
from tensorflow.python.training import saver
from tensorflow.python.training.session_manager import SessionManager
from torchvision.transforms import transforms

from nninst import AttrMap, Graph, GraphAttrKey, mode
from nninst.backend.pytorch.model import LeNet as TorchLeNet
from nninst.backend.tensorflow.attack.common import (
    calculate_overlap,
    get_overlay_summary,
)
from nninst.backend.tensorflow.dataset import mnist
from nninst.backend.tensorflow.graph import model_fn_with_fetch_hook
from nninst.backend.tensorflow.model import LeNet
from nninst.backend.tensorflow.trace import (
    lenet_mnist_class_trace,
    reconstruct_trace_from_tf,
)
from nninst.backend.tensorflow.utils import new_session_config
from nninst.dataset import mnist_info
from nninst.trace import TraceKey
from nninst.utils.fs import CsvIOAction, IOAction, abspath
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init, ray_iter


def overlap_ratio(
    attack_fn,
    generate_adversarial_fn,
    class_trace_fn: Callable[[int], IOAction[AttrMap]],
    select_fn: Callable[[np.ndarray], np.ndarray],
    overlap_fn: Callable[[AttrMap, AttrMap, Graph], Dict[str, Any]],
    path: str,
    **kwargs,
):
    def get_overlap_ratio() -> pd.DataFrame:
        def get_row(image_id: int) -> dict:
            print(image_id)
            data_dir = abspath("/home/yxqiu/data/mnist/raw")
            model_dir = abspath("tf/lenet/model/")
            create_model = lambda: LeNet(data_format="channels_first")
            model_fn = partial(model_fn_with_fetch_hook, create_model=create_model)

            trace = reconstruct_trace_from_tf(
                model_fn=model_fn,
                input_fn=lambda: mnist.test(data_dir).skip(image_id).take(1).batch(1),
                select_fn=select_fn,
                model_dir=model_dir,
            )[0]

            def torch_to_tf_trace(torch_trace, debug: bool = False):
                torch_to_tf_mapping = {
                    "conv_1": "conv2d/Conv2D",
                    "pool_1": "max_pooling2d/MaxPool",
                    "conv_2": "conv2d_1/Conv2D",
                    "pool_2": "max_pooling2d/MaxPool_1",
                    "linear_3": "dense/MatMul",
                    "linear_4": "dense_1/MatMul",
                    "linear_5": "dense_2/MatMul",
                }
                new_trace = AttrMap()
                if debug:
                    new_trace.ops = {
                        torch_to_tf_mapping[name]: {
                            TraceKey.EDGE: torch_trace[name][0],
                            TraceKey.WEIGHTED_INPUT: torch_trace[name][1],
                        }
                        for name in torch_to_tf_mapping
                    }
                else:
                    new_trace.ops = {
                        torch_to_tf_mapping[name]: {TraceKey.EDGE: torch_trace[name]}
                        for name in torch_to_tf_mapping
                    }
                return new_trace

            analysis = LenetAnalysis()
            infer_info = analysis.infer_info(image_id, is_val=True)
            torch_trace = torch_to_tf_trace(
                infer_info.traced_edges_by_layer(threshold, arg_approx)
            )

            def compare_trace(
                tf_trace: AttrMap, torch_trace: AttrMap, debug: bool = False
            ):
                for name, attrs in torch_trace.ops.items():
                    tf_attrs = tf_trace.ops[name]
                    tf_edge = np.sort(tf_attrs[TraceKey.EDGE])
                    torch_edge = np.sort(attrs[TraceKey.EDGE])
                    if not np.all(tf_edge == torch_edge):
                        print(f"layer {name}")
                        if debug:
                            print(
                                "tf+: {}".format(
                                    {
                                        diff: tf_attrs[TraceKey.WEIGHTED_INPUT][
                                            np.where(tf_edge == diff)
                                        ][0]
                                        for diff in np.setdiff1d(tf_edge, torch_edge)
                                    }
                                )
                            )
                            print(
                                "tf-: {}".format(
                                    {
                                        diff: attrs[TraceKey.WEIGHTED_INPUT][
                                            np.where(torch_edge == diff)
                                        ][0]
                                        for diff in np.setdiff1d(torch_edge, tf_edge)
                                    }
                                )
                            )
                        else:
                            print("tf+: {}".format(np.setdiff1d(tf_edge, torch_edge)))
                            print("tf-: {}".format(np.setdiff1d(torch_edge, tf_edge)))
                        break

            compare_trace(trace, torch_trace)

            label = mnist_info.test().label(image_id)

            if (label != trace.attrs[GraphAttrKey.PREDICT]) or (
                label != infer_info.predict
            ):
                return {}

            adversarial_example = generate_adversarial_fn(
                label=label,
                create_model=create_model,
                input_fn=lambda: mnist.test(data_dir, normed=False)
                .skip(image_id)
                .take(1)
                .batch(1)
                .make_one_shot_iterator()
                .get_next()[0],
                attack_fn=attack_fn,
                model_dir=model_dir,
                image_id=image_id,
                **kwargs,
            )

            if adversarial_example is None:
                return {}

            adversarial_trace, adversarial_graph = reconstruct_trace_from_tf(
                model_fn=model_fn,
                input_fn=lambda: tf.data.Dataset.from_tensors(
                    mnist.normalize(adversarial_example)
                ),
                select_fn=select_fn,
                model_dir=model_dir,
                debug=True,
            )[0]

            # torch_adversarial_example = functional.to_pil_image(torch.from_numpy(adversarial_example[0].copy()))
            adversarial_infer_info = analysis.infer_info(
                image_id, is_val=True, image=adversarial_example[0].copy()
            )
            torch_adversarial_trace = torch_to_tf_trace(
                adversarial_infer_info.traced_edges_by_layer(
                    threshold, arg_approx, debug=True
                ),
                debug=True,
            )

            compare_trace(adversarial_trace, torch_adversarial_trace, debug=True)

            if adversarial_graph.attrs["predict"] == label:
                print("attack fail on tensorflow")

            if adversarial_infer_info.predict == label:
                print("attack fail on pytorch")

            normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            # transform = transforms.Compose([transforms.ToTensor(), normalize])
            transform = transforms.Compose([normalize])
            # torch_adversarial_example_np = (transform(
            #     functional.to_pil_image(torch.from_numpy(adversarial_example[0].copy())))
            #                                 .unsqueeze(0).cpu().numpy())
            torch_adversarial_example_np = (
                transform(torch.from_numpy(adversarial_example[0].copy()))
                .unsqueeze(0)
                .cpu()
                .numpy()
            )
            tf_adversarial_example_np = mnist.normalize(adversarial_example)
            assert np.all(torch_adversarial_example_np == tf_adversarial_example_np)

            adversarial_label = adversarial_trace.attrs[GraphAttrKey.PREDICT]

            if label != adversarial_label:

                def map_prefix(map: Dict[str, Any], prefix: str) -> Dict[str, Any]:
                    return {f"{prefix}.{key}": value for key, value in map.items()}

                adversarial_graph = LeNet.create_graph()
                row = {
                    "image_id": image_id,
                    **map_prefix(
                        overlap_fn(
                            class_trace_fn(label).load(), trace, adversarial_graph
                        ),
                        "original",
                    ),
                    **map_prefix(
                        overlap_fn(
                            class_trace_fn(adversarial_label).load(),
                            adversarial_trace,
                            adversarial_graph,
                        ),
                        "adversarial",
                    ),
                }

                def overlay_ratio(class_id: int, image_id: int, image: np.ndarray):
                    # image = F.to_pil_image(torch.from_numpy(image))
                    infer_info = analysis.infer_info(image_id, is_val=True, image=image)
                    original_traces = {
                        layer_name: (
                            get_class_per_layer_traced_edges(
                                class_id, threshold, layer_name, label="best_in_10"
                            ).index.get_values()
                        )
                        for layer_name in analysis.lenet.weighted_layer_names
                    }
                    traces = infer_info.traced_edges_by_layer(
                        threshold, input_filter=arg_approx
                    )
                    trace_num = sum(
                        len(traces[layer_name])
                        for layer_name in analysis.lenet.weighted_layer_names
                    )
                    return (
                        sum(
                            len(
                                np.intersect1d(
                                    traces[layer_name], original_traces[layer_name]
                                )
                            )
                            for layer_name in analysis.lenet.weighted_layer_names
                        )
                        / trace_num
                    )

                original_overlay_ratio = overlay_ratio(label, image_id, None)
                adversarial_overlap_ratio = overlay_ratio(
                    adversarial_label, image_id, adversarial_example[0].copy()
                )
                assert original_overlay_ratio == row["original.trace.edge"]
                # assert adversarial_overlap_ratio == row["adversarial.trace.edge"]
                return row
            else:
                return {}

        # traces = ray_iter(get_row, (image_id for image_id in range(300, 350)),
        # traces = ray_iter(get_row, (image_id for image_id in range(131, 300)),
        traces = ray_iter(
            get_row,
            (image_id for image_id in range(mnist_info.test().size)),
            chunksize=1,
            out_of_order=True,
            num_gpus=1,
        )
        # chunksize=1, out_of_order=False, num_gpus=1)
        # count = 0
        # result = []
        # for trace in traces:
        #     result.append(trace)
        #     print(count)
        #     count += 1
        # traces = [trace for trace in result if len(trace) != 0]
        traces = [trace for trace in traces if len(trace) != 0]
        return pd.DataFrame(traces)

    return CsvIOAction(path, init_fn=get_overlap_ratio)


def generate_adversarial_example(
    label: int,
    create_model,
    input_fn: Callable[[], tf.Tensor],
    attack_fn: Callable[..., Attack],
    model_dir=None,
    checkpoint_path=None,
    preprocessing=(0, 1),
    image_id=None,
    **kwargs,
) -> Optional[np.ndarray]:
    # Check that model has been trained.
    if not checkpoint_path:
        checkpoint_path = saver.latest_checkpoint(model_dir)
    if not checkpoint_path:
        raise ValueError(
            "Could not find trained model in model_dir: {}.".format(model_dir)
        )

    with tf.Graph().as_default():
        features = input_fn()
        model = create_model()
        image_tensor = tf.placeholder(features.dtype, features.shape)
        logits = model(image_tensor)

        torch_model = TorchLeNet()
        path = abspath("lenet_model.pth")
        torch_model.load_state_dict(torch.load(path))
        torch_model.cuda()
        device_id = torch.cuda.current_device()
        torch_model.eval()
        attack_model = foolbox.models.PyTorchModel(
            torch_model,
            bounds=(0, 1),
            num_classes=10,
            channel_axis=1,
            preprocessing=preprocessing,
        )
        sm = SessionManager()
        with sm.prepare_session(
            master="",
            saver=tf.train.Saver(),
            checkpoint_filename_with_path=checkpoint_path,
            config=new_session_config(),
        ) as sess:
            image = sess.run(features)[0]
            # attack_model = TensorFlowModel(
            #     image_tensor,
            #     logits,
            #     bounds=(0, 1),
            #     channel_axis=1,
            #     preprocessing=preprocessing)

            # image, _ = torch_mnist.image(image_id, is_val=True)
            # transform = transforms.Compose([transforms.ToTensor()])
            # image = transform(image).cpu().numpy()
            attack = attack_fn(attack_model)
            with torch.cuda.device(device_id):
                adversarial_example = attack(image, label=label, **kwargs)
            # adversarial_example = attack(image[0], label=label, **kwargs)
            if adversarial_example is None:
                return None
            else:
                return adversarial_example[np.newaxis]


if __name__ == "__main__":
    mode.debug()
    # mode.distributed()
    ray_init()
    threshold = 0.5
    # threshold = 1
    # threshold = 0.8
    attacks = {
        "FGSM": [FGSM],
        "BIM": [IterativeGradientSignAttack],
        "JSMA": [SaliencyMapAttack],
        "DeepFool": [DeepFoolAttack],
        "DeepFool_full": [DeepFoolAttack, dict(subsample=None)],
    }

    for attack_name in [
        # "FGSM",
        # "BIM",
        # "JSMA",
        "DeepFool",
        # "DeepFool_full",
    ]:
        # label = "early"
        # label = "best_in_10"
        # label = "worst_in_10"
        # label = "import"
        label = "norm"
        lenet_overlap_ratio = overlap_ratio(
            attack_fn=attacks[attack_name][0],
            generate_adversarial_fn=generate_adversarial_example,
            class_trace_fn=lambda class_id: lenet_mnist_class_trace(
                class_id, threshold, label=label
            ),
            # class_trace_fn=lambda class_id: lenet_mnist_class_trace(class_id, threshold),
            select_fn=lambda input: arg_approx(input, threshold),
            overlap_fn=calculate_overlap,
            path="lenet_class_overlap_ratio_{0:.1f}_{1}_{2}.foolbox.csv".format(
                threshold, attack_name, label
            ),
            # path='lenet_class_overlap_ratio_{:.1f}_{}.foolbox.csv'.format(threshold, attack_name),
            preprocessing=(0.1307, 0.3081),
            **(attacks[attack_name][1] if len(attacks[attack_name]) == 2 else {}),
        )

        lenet_overlap_ratio.save()
        print(f"attack: {attack_name}")
        print("edge:")
        print(get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.EDGE))
        print("weight:")
        print(get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.WEIGHT))
        print("point:")
        print(get_overlay_summary(lenet_overlap_ratio.load(), TraceKey.POINT))
