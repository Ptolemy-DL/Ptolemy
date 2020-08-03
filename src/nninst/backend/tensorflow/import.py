import os
from functools import partial
from typing import Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from lenet_analysis import LenetAnalysis, get_class_per_layer_traced_edges
from PIL import Image
from torch.autograd import Variable
from torch.nn import Module
from torchvision.models import AlexNet
from torchvision.transforms import transforms

from nninst import mode
from nninst.backend.pytorch.model import LeNet as TorchLeNet
from nninst.backend.tensorflow.dataset import mnist
from nninst.backend.tensorflow.graph import model_fn_with_fetch_hook
from nninst.backend.tensorflow.model import AlexNet as TFAlexNet
from nninst.backend.tensorflow.model import LeNet as TFLeNet
from nninst.backend.tensorflow.trace import (
    lenet_mnist_class_trace,
    reconstruct_trace_from_tf,
)
from nninst.backend.tensorflow.utils import new_session_config
from nninst.dataset import mnist_info
from nninst.utils.fs import CsvIOAction, abspath
from nninst.utils.numpy import arg_approx
from nninst.utils.ray import ray_init, ray_map

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def iterate_model(module: Module, structure: Dict, action, *args, **kwargs):
    children = dict(module.named_modules())
    for key, value in structure.items():
        child = children[str(key)]
        if isinstance(value, str):
            name = value
            action(name, child, *args, **kwargs)
        elif isinstance(value, dict):
            iterate_model(child, value, action, *args, **kwargs)
        else:
            raise TypeError(f"value: {value}, type: {type(value)}")


def to_numpy(tensor) -> np.ndarray:
    return tensor.data.cpu().numpy()


def import_model_from_pytorch():
    model = TorchLeNet()
    path = abspath("lenet_model.pth")
    model.load_state_dict(torch.load(path))
    structure = {
        "features": {
            "0": "conv_1",
            "1": "relu_1",
            "2": "pool_1",
            "3": "conv_2",
            "4": "relu_2",
            "5": "pool_2",
        },
        "classifier": {
            "0": "linear_3",
            "1": "relu_3",
            "2": "linear_4",
            "3": "relu_4",
            "4": "linear_5",
        },
    }
    tf_to_torch_variables = {
        "conv2d/kernel:0": "conv_1/weight",
        "conv2d/bias:0": "conv_1/bias",
        "conv2d_1/kernel:0": "conv_2/weight",
        "conv2d_1/bias:0": "conv_2/bias",
        "dense/kernel:0": "linear_3/weight",
        "dense/bias:0": "linear_3/bias",
        "dense_1/kernel:0": "linear_4/weight",
        "dense_1/bias:0": "linear_4/bias",
        "dense_2/kernel:0": "linear_5/weight",
        "dense_2/bias:0": "linear_5/bias",
    }

    def fetch_variables(name: str, layer: Module):
        if "linear" in name or "conv" in name:
            if "conv" in name:
                variables[f"{name}/weight"] = np.transpose(
                    to_numpy(layer.weight), (2, 3, 1, 0)
                )
            else:
                variables[f"{name}/weight"] = np.transpose(
                    to_numpy(layer.weight), (1, 0)
                )
            variables[f"{name}/bias"] = to_numpy(layer.bias)

    variables = {}
    iterate_model(model, structure, fetch_variables)
    lenet = TFLeNet()
    input_tensor = tf.placeholder(tf.float32, (None, 1, 28, 28))
    lenet(input_tensor)
    tf.train.create_global_step()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for variable in tf.global_variables():
            if variable.name != "global_step:0":
                variable.load(
                    variables[tf_to_torch_variables[variable.name]], session=sess
                )
        saver = tf.train.Saver()
        saver.save(sess, abspath("tf/lenet/model_import/model"))


def import_alexnet_model_from_pytorch():
    model = AlexNet()
    path = abspath("cache/alexnet-owt-4df8aa71.pth")
    model.load_state_dict(torch.load(path))
    structure = {
        "features": {
            "0": "conv_1",
            "1": "relu_1",
            "2": "pool_1",
            "3": "conv_2",
            "4": "relu_2",
            "5": "pool_2",
            "6": "conv_3",
            "7": "relu_3",
            "8": "conv_4",
            "9": "relu_4",
            "10": "conv_5",
            "11": "relu_5",
            "12": "pool_5",
        },
        "classifier": {
            "0": "dropout_6",
            "1": "linear_6",
            "2": "relu_6",
            "3": "dropout_7",
            "4": "linear_7",
            "5": "relu_7",
            "6": "linear_8",
        },
    }
    tf_to_torch_variables = {
        "conv2d/kernel:0": "conv_1/weight",
        "conv2d/bias:0": "conv_1/bias",
        "conv2d_1/kernel:0": "conv_2/weight",
        "conv2d_1/bias:0": "conv_2/bias",
        "conv2d_2/kernel:0": "conv_3/weight",
        "conv2d_2/bias:0": "conv_3/bias",
        "conv2d_3/kernel:0": "conv_4/weight",
        "conv2d_3/bias:0": "conv_4/bias",
        "conv2d_4/kernel:0": "conv_5/weight",
        "conv2d_4/bias:0": "conv_5/bias",
        "dense/kernel:0": "linear_6/weight",
        "dense/bias:0": "linear_6/bias",
        "dense_1/kernel:0": "linear_7/weight",
        "dense_1/bias:0": "linear_7/bias",
        "dense_2/kernel:0": "linear_8/weight",
        "dense_2/bias:0": "linear_8/bias",
    }

    def fetch_variables(name: str, layer: Module):
        if "linear" in name or "conv" in name:
            if "conv" in name:
                variables[f"{name}/weight"] = np.transpose(
                    to_numpy(layer.weight), (2, 3, 1, 0)
                )
            else:
                variables[f"{name}/weight"] = np.transpose(
                    to_numpy(layer.weight), (1, 0)
                )
            variables[f"{name}/bias"] = to_numpy(layer.bias)

    variables = {}
    iterate_model(model, structure, fetch_variables)
    alexnet = TFAlexNet()
    input_tensor = tf.placeholder(tf.float32, (None, 224, 224, 3))
    alexnet(input_tensor)
    tf.train.create_global_step()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for variable in tf.global_variables():
            if variable.name != "global_step:0":
                variable.load(
                    variables[tf_to_torch_variables[variable.name]], session=sess
                )
        saver = tf.train.Saver()
        saver.save(sess, abspath("tf/alexnet/model_import/model"))


def lenet_model_fn(features, labels, mode):
    """The model_fn argument for creating an Estimator."""
    model = TFLeNet()
    image = features
    if isinstance(image, dict):
        image = features["image"]

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(image, training=False)
        predictions = {
            "predict": tf.argmax(logits, axis=1),
            "logit": tf.reduce_max(logits, axis=1),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={"classify": tf.estimator.export.PredictOutput(predictions)},
        )


def compare_predict():
    def get_predict() -> pd.DataFrame:
        def get_row(image_id: int) -> dict:
            data_dir = abspath("/home/yxqiu/data/mnist/raw")
            model_dir = abspath("tf/lenet/model_import/")
            # model_dir = abspath("tf/lenet/model/")
            estimator_config = tf.estimator.RunConfig(
                session_config=new_session_config()
            )
            classifier = tf.estimator.Estimator(
                model_fn=lenet_model_fn, model_dir=model_dir, config=estimator_config
            )

            predictions = list(
                classifier.predict(
                    input_fn=lambda: mnist.test(data_dir)
                    .skip(image_id)
                    .take(1)
                    .batch(1)
                )
            )

            def get_image_val(image: Image) -> Variable:
                normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                transform = transforms.Compose([transforms.ToTensor(), normalize])
                tensor = transform(image).unsqueeze(0)
                image_val = Variable(tensor, volatile=True)
                return image_val

            image, label = mnist_info.test().image_with_label(image_id)
            model = TorchLeNet()
            path = abspath("lenet_model.pth")
            model.load_state_dict(torch.load(path))
            image_val = get_image_val(image)
            model.eval()
            result = model(image_val).data.cpu().numpy()

            def map_prefix(map: Dict[str, Any], prefix: str) -> Dict[str, Any]:
                return {f"{prefix}.{key}": value for key, value in map.items()}

            row = {
                "image_id": image_id,
                "label": label,
                **map_prefix(predictions[0], "tf"),
                **map_prefix(
                    {"predict": result.argmax(), "logit": result.max()}, "pytorch"
                ),
            }
            return row

        mode.debug()
        # mode.distributed()
        ray_init()
        # traces = ray_map(get_row, (image_id for image_id in range(mnist_info.test().size)),
        traces = ray_map(
            get_row,
            (image_id for image_id in range(10)),
            chunksize=1,
            out_of_order=True,
            num_gpus=1,
        )
        return pd.DataFrame(traces)

    return CsvIOAction("compare_predict.csv", init_fn=get_predict)


def compare_trace():
    def get_trace() -> pd.DataFrame:
        def get_row(image_id: int) -> dict:
            threshold = 0.5
            data_dir = abspath("/home/yxqiu/data/mnist/raw")
            # model_dir = abspath("tf/lenet/model_import/")
            model_dir = abspath("tf/lenet/model/")

            model_fn = partial(
                model_fn_with_fetch_hook,
                create_model=lambda: TFLeNet(data_format="channels_first"),
            )

            tf_trace = reconstruct_trace_from_tf(
                model_fn=model_fn,
                input_fn=lambda: mnist.train(data_dir).skip(image_id).take(1).batch(1),
                select_fn=lambda input: arg_approx(input, threshold),
                model_dir=model_dir,
            )[0]

            def get_image_val(image: Image) -> Variable:
                normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                transform = transforms.Compose([transforms.ToTensor(), normalize])
                tensor = transform(image).unsqueeze(0)
                image_val = Variable(tensor, volatile=True)
                return image_val

            analysis = LenetAnalysis()
            pytorch_trace = analysis.infer_info(image_id).layer_traces(
                threshold, arg_approx
            )
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

            def map_prefix(map: Dict[str, Any], prefix: str) -> Dict[str, Any]:
                return {f"{prefix}.{key}": value for key, value in map.items()}

            row = {"image_id": image_id}
            return row

        mode.debug()
        # mode.distributed()
        ray_init()
        # traces = ray_map(get_row, (image_id for image_id in range(mnist_info.train().size)),
        traces = ray_map(
            get_row,
            (image_id for image_id in range(10)),
            chunksize=1,
            out_of_order=True,
            num_gpus=1,
        )
        return pd.DataFrame(traces)

    return CsvIOAction("compare_trace.csv", init_fn=get_trace)


def compare_class_trace():
    def get_trace() -> pd.DataFrame:
        def get_row(class_id: int) -> dict:
            threshold = 0.5
            tf_label = "norm"
            torch_label = "best_in_10"

            tf_trace = lenet_mnist_class_trace(
                class_id, threshold, label=tf_label
            ).load()

            analysis = LenetAnalysis()
            torch_trace = {
                layer_name: (
                    get_class_per_layer_traced_edges(
                        class_id, threshold, layer_name, label=torch_label
                    ).index.get_values()
                )
                for layer_name in analysis.lenet.weighted_layer_names
            }
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

            row = {}
            return row

        mode.debug()
        # mode.distributed()
        ray_init()
        # traces = ray_map(get_row, (image_id for image_id in range(mnist_info.train().size)),
        traces = ray_map(
            get_row,
            (class_id for class_id in range(10)),
            chunksize=1,
            out_of_order=True,
            num_gpus=1,
        )
        return pd.DataFrame(traces)

    return CsvIOAction("compare_class_trace.csv", init_fn=get_trace)


if __name__ == "__main__":
    # import_model_from_pytorch()
    # compare_predict().save()
    # compare_trace().save()
    # compare_class_trace().save()
    import_alexnet_model_from_pytorch()
