import shutil
import tempfile
import traceback
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, Iterable, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import assign_from_checkpoint_fn
from tensorflow.python.estimator.estimator import WarmStartSettings
from tensorflow.python.training.session_run_hook import SessionRunArgs, SessionRunHook

from nninst.backend.tensorflow.attack.common import imagenet_example
from nninst.backend.tensorflow.dataset import imagenet_raw
from nninst.backend.tensorflow.dataset.config import CIFAR100_TEST
from nninst.backend.tensorflow.graph import model_fn_with_fetch_hook
from nninst.backend.tensorflow.model.config import (
    ALEXNET,
    ALEXNET_CDRP,
    RESNET_18_CIFAR100_CDRP,
    RESNET_50,
    RESNET_50_CDRP,
    VGG_16,
    VGG_16_CDRP,
    ModelConfig,
)
from nninst.backend.tensorflow.trace import predict, reconstruct_trace_from_tf
from nninst.backend.tensorflow.utils import new_session_config
from nninst.dataset.envs import IMAGENET_RAW_DIR
from nninst.utils import merge_dict
from nninst.utils.fs import IOAction, abspath
from nninst.utils.ray import ray_iter


def generate_example_cdrp(
    example_trace_fn: Callable[..., IOAction[Dict[str, np.ndarray]]],
    class_ids: Iterable[int],
    image_ids: Iterable[int],
    attack_name: str,
    attack_fn,
    generate_adversarial_fn,
    cache: bool = True,
    batch_size: int = 1,
    **kwargs,
):
    def generate_examples_fn(
        class_id: int, image_id: int
    ) -> Union[Tuple[int, int], Tuple[int, int, str]]:
        try:
            class_id = int(class_id)
            image_id = int(image_id)
            example_trace_io = example_trace_fn(
                attack_name=attack_name,
                attack_fn=attack_fn,
                generate_adversarial_fn=generate_adversarial_fn,
                class_id=class_id,
                image_id=image_id,
                cache=cache,
                batch_size=batch_size,
            )
            example_trace_io.save()
            return class_id, image_id
        except Exception:
            return class_id, image_id, traceback.format_exc()

    print(f"begin {attack_name}")
    results = ray_iter(
        generate_examples_fn,
        [(class_id, image_id) for image_id in image_ids for class_id in class_ids],
        chunksize=1,
        out_of_order=True,
        num_gpus=0,
        huge_task=True,
    )
    for result in results:
        if len(result) == 3:
            class_id, image_id, tb = result
            print(f"## raise exception from class {class_id}, image {image_id}:")
            print(tb)
        else:
            class_id, image_id = result
            # print(f"finish class {class_id} image {image_id}")
    print(f"finish {attack_name}")


def alexnet_imagenet_example_cdrp(
    attack_name,
    attack_fn,
    generate_adversarial_fn,
    class_id: int,
    image_id: int,
    cache: bool = True,
    batch_size: int = 1,
) -> IOAction[Dict[str, np.ndarray]]:
    return imagenet_example_cdrp(
        model_config=ALEXNET_CDRP.with_model_dir("tf/alexnet/model_import"),
        attack_name=attack_name,
        attack_fn=attack_fn,
        generate_adversarial_fn=generate_adversarial_fn,
        class_id=class_id,
        image_id=image_id,
        cache=cache,
        batch_size=batch_size,
    )


def resnet_18_cifar100_example_cdrp(
    attack_name,
    attack_fn,
    generate_adversarial_fn,
    class_id: int,
    image_id: int,
    cache: bool = True,
    batch_size: int = 1,
) -> IOAction[Dict[str, np.ndarray]]:
    return cifar100_example_cdrp(
        model_config=RESNET_18_CIFAR100_CDRP,
        attack_name=attack_name,
        attack_fn=attack_fn,
        generate_adversarial_fn=generate_adversarial_fn,
        class_id=class_id,
        image_id=image_id,
        cache=cache,
        batch_size=batch_size,
    )


def resnet_50_imagenet_example_cdrp(
    attack_name,
    attack_fn,
    generate_adversarial_fn,
    class_id: int,
    image_id: int,
    cache: bool = True,
) -> IOAction[Dict[str, np.ndarray]]:
    return imagenet_example_cdrp(
        model_config=RESNET_50_CDRP,
        attack_name=attack_name,
        attack_fn=attack_fn,
        generate_adversarial_fn=generate_adversarial_fn,
        class_id=class_id,
        image_id=image_id,
        cache=cache,
    )


def vgg_16_imagenet_example_cdrp(
    attack_name,
    attack_fn,
    generate_adversarial_fn,
    class_id: int,
    image_id: int,
    cache: bool = True,
) -> IOAction[Dict[str, np.ndarray]]:
    return imagenet_example_cdrp(
        model_config=VGG_16_CDRP,
        attack_name=attack_name,
        attack_fn=attack_fn,
        generate_adversarial_fn=generate_adversarial_fn,
        class_id=class_id,
        image_id=image_id,
        cache=cache,
    )


def imagenet_example_cdrp(
    model_config: ModelConfig,
    attack_name,
    attack_fn,
    generate_adversarial_fn,
    class_id: int,
    image_id: int,
    cache: bool = True,
    batch_size: int = 1,
) -> IOAction[Dict[str, np.ndarray]]:
    def get_example_cdrp() -> Dict[str, np.ndarray]:
        data_dir = IMAGENET_RAW_DIR
        model_dir = abspath(model_config.model_dir)
        create_original_model = lambda: model_config.network_class(with_gates=False)
        create_model = lambda: model_config.network_class()
        input_fn = lambda: imagenet_raw.test(
            data_dir,
            class_id,
            image_id,
            class_from_zero=model_config.class_from_zero,
            preprocessing_fn=model_config.preprocessing_fn,
        )
        batched_input_fn = lambda: imagenet_raw.test(
            data_dir,
            class_id,
            image_id,
            class_from_zero=model_config.class_from_zero,
            preprocessing_fn=model_config.preprocessing_fn,
            batch=batch_size,
        ).batch(batch_size)
        predicted_label = predict(
            create_model=create_original_model, input_fn=input_fn, model_dir=model_dir
        )

        if predicted_label != class_id:
            return None

        if attack_name == "original":
            trace = reconstruct_cdrp_from_tf(
                create_model=create_model,
                class_id=class_id,
                input_fn=batched_input_fn,
                model_dir=model_dir,
                batch_size=batch_size,
            )
            return trace

        adversarial_example = imagenet_example(
            model_config=model_config,
            attack_name=attack_name,
            attack_fn=attack_fn,
            generate_adversarial_fn=generate_adversarial_fn,
            class_id=class_id,
            image_id=image_id,
        ).load()

        if adversarial_example is None:
            return None

        adversarial_input_fn = lambda: tf.data.Dataset.from_tensors(
            model_config.normalize_fn(adversarial_example)
        )
        adversarial_predicted_label = predict(
            create_model=create_original_model,
            input_fn=adversarial_input_fn,
            model_dir=model_dir,
        )

        if predicted_label == adversarial_predicted_label:
            return None

        if batch_size > 1:
            batched_adversarial_input_fn = (
                lambda: tf.data.Dataset.from_tensors(
                    (
                        model_config.normalize_fn(adversarial_example)[0],
                        adversarial_predicted_label,
                    )
                )
                .concatenate(
                    imagenet_raw.test(
                        data_dir,
                        adversarial_predicted_label,
                        image_id + 1,
                        class_from_zero=model_config.class_from_zero,
                        preprocessing_fn=model_config.preprocessing_fn,
                        batch=batch_size - 1,
                    )
                )
                .batch(batch_size)
            )
        else:
            batched_adversarial_input_fn = adversarial_input_fn

        adversarial_trace = reconstruct_cdrp_from_tf(
            create_model=create_model,
            input_fn=batched_adversarial_input_fn,
            model_dir=model_dir,
            class_id=adversarial_predicted_label,
            batch_size=batch_size,
        )
        return adversarial_trace

    name = f"{model_config.name}_imagenet"
    trace_name = "example_cdrp"
    if batch_size == 1:
        path = f"store/{trace_name}/{attack_name}/{name}/{class_id}/{image_id}.pkl"
    else:
        path = f"store/{trace_name}/{attack_name}/{name}/batch={batch_size}/{class_id}/{image_id}.pkl"
    return IOAction(path, init_fn=get_example_cdrp, cache=cache, compress=True)


def cifar100_example_cdrp(
    model_config: ModelConfig,
    attack_name,
    attack_fn,
    generate_adversarial_fn,
    class_id: int,
    image_id: int,
    cache: bool = True,
    batch_size: int = 1,
) -> IOAction[Dict[str, np.ndarray]]:
    assert batch_size == 1

    def get_example_cdrp() -> Dict[str, np.ndarray]:
        data_config = CIFAR100_TEST
        model_dir = abspath(model_config.model_dir)
        create_original_model = lambda: model_config.network_class(with_gates=False)
        create_model = lambda: model_config.network_class()
        input_fn = lambda: data_config.dataset_fn(
            data_config.data_dir,
            batch_size=1,
            transform_fn=lambda dataset: dataset.filter(
                lambda image, label: tf.equal(
                    tf.convert_to_tensor(class_id, dtype=tf.int32), label
                )
            )
            .skip(image_id)
            .take(batch_size),
        )
        batched_input_fn = lambda: data_config.dataset_fn(
            data_config.data_dir,
            batch_size=1,
            transform_fn=lambda dataset: dataset.filter(
                lambda image, label: tf.equal(
                    tf.convert_to_tensor(class_id, dtype=tf.int32), label
                )
            )
            .skip(image_id)
            .take(batch_size),
        )
        predicted_label = predict(
            create_model=create_original_model, input_fn=input_fn, model_dir=model_dir
        )

        if predicted_label != class_id:
            return None

        if attack_name == "original":
            trace = reconstruct_cdrp_from_tf(
                create_model=create_model,
                class_id=class_id,
                input_fn=batched_input_fn,
                model_dir=model_dir,
                batch_size=batch_size,
            )
            return trace

        adversarial_example = imagenet_example(
            model_config=model_config,
            attack_name=attack_name,
            attack_fn=attack_fn,
            generate_adversarial_fn=generate_adversarial_fn,
            class_id=class_id,
            image_id=image_id,
        ).load()

        if adversarial_example is None:
            return None

        adversarial_input_fn = lambda: tf.data.Dataset.from_tensors(
            model_config.normalize_fn(adversarial_example)
        )
        adversarial_predicted_label = predict(
            create_model=create_original_model,
            input_fn=adversarial_input_fn,
            model_dir=model_dir,
        )

        if predicted_label == adversarial_predicted_label:
            return None

        batched_adversarial_input_fn = adversarial_input_fn

        adversarial_trace = reconstruct_cdrp_from_tf(
            create_model=create_model,
            input_fn=batched_adversarial_input_fn,
            model_dir=model_dir,
            class_id=adversarial_predicted_label,
            batch_size=batch_size,
        )
        return adversarial_trace

    name = f"{model_config.name}"
    trace_name = "example_cdrp"
    if batch_size == 1:
        path = f"store/{trace_name}/{attack_name}/{name}/{class_id}/{image_id}.pkl"
    else:
        path = f"store/{trace_name}/{attack_name}/{name}/batch={batch_size}/{class_id}/{image_id}.pkl"
    return IOAction(path, init_fn=get_example_cdrp, cache=cache, compress=True)


class Variables:
    def __init__(self):
        self.original_predictions: np.ndarray = None
        self.original_logits: np.ndarray = None
        self.gates: Dict[str, np.ndarray] = None
        self.prev_gates: Dict[str, np.ndarray] = None
        self.gate_variables: Dict[str, tf.Variable] = None
        self.prev_loss: float = None
        self.loss: float = None
        self.last_gates = None
        self.correct = True


class FetchGatesHook(SessionRunHook):
    def __init__(self, variables: Variables, fetches, class_id):
        self.variables = variables
        self.fetches = fetches
        self.class_id = class_id
        self.loss = None
        self.canceled = False
        self.used_before_run = False
        self.used_after_run = False
        self.used_end = False

    def before_run(self, run_context):
        if self.used_before_run:
            return None
        self.used_before_run = True
        if self.variables.gates is None:
            fetches = merge_dict(self.variables.gate_variables, self.fetches)
        else:
            fetches = self.fetches
            session = run_context.session
            for name, gate_variable in self.variables.gate_variables.items():
                gate_variable.load(self.variables.gates[name], session)
        return SessionRunArgs(fetches=fetches)

    def after_run(self, run_context, run_values):
        if self.used_after_run:
            return None
        self.used_after_run = True
        if self.variables.gates is None:
            self.variables.gates = {
                name: run_values.results[name] for name in self.variables.gate_variables
            }
        else:
            predictions = run_values.results["predictions"]
            if np.argmax(predictions[0]) != self.class_id:
                self.variables.correct = False
                if self.variables.last_gates is None:
                    self.variables.last_gates = self.variables.prev_gates
                # self.variables.loss = self.variables.prev_loss
                # self.variables.gates = self.variables.prev_gates
                # self.canceled = True
            else:
                self.variables.correct = True
                if self.variables.last_gates is not None:
                    self.variables.last_gates = None
        self.loss = run_values.results["loss"]
        # loss = self.loss
        # l1_loss = run_values.results["l1_loss"]
        # cross_entropy = run_values.results["cross_entropy"]
        # gradients = run_values.results["gradient"]
        # global_step = run_values.results["global_step"]

    def end(self, session):
        if self.used_end:
            return None
        self.used_end = True
        if (not self.canceled) and (
            self.variables.loss is None or self.variables.loss > self.loss
        ):
            gates = session.run(self.variables.gate_variables)
            # gates = {name: np.clip(gate - 0.1 * 0.05 * np.sign(gate), 0, 10) for name, gate in gates.items()}
            # if np.any([np.any(np.logical_or(gate < 0, gate > 10)) for gate in gates.values()]):
            #     pass
            gates = {name: np.clip(gate, 0, 10) for name, gate in gates.items()}
            # gates = {name: np.clip(gate - self.lr * self.penalty * np.sign(self.variables.gates[name]), 0, 10)
            #          for name, gate in gates.items()}
            self.variables.prev_loss = self.variables.loss
            self.variables.prev_gates = self.variables.gates
            self.variables.loss = self.loss
            self.variables.gates = gates


def model_fn(create_model, cdrp_variables: Variables, features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    model = create_model()
    penalty = 0.05
    lr = 0.1
    image = features
    class_id = params["class_id"]
    batch_size = params["batch_size"]
    if isinstance(image, dict):
        image = features["image"]
    cdrp_variables.gate_variables = OrderedDict()
    logits = model(
        image,
        gate_variables=cdrp_variables.gate_variables,
        batch_size=batch_size,
        training=False,
    )

    global_step = tf.train.get_or_create_global_step()
    softmax = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.convert_to_tensor(cdrp_variables.original_predictions),
                logits=logits,
            )
        )
        l1_loss = tf.add_n(
            [tf.reduce_sum(gate) for gate in cdrp_variables.gate_variables.values()]
        )
        # l1_loss = tf.add_n([tf.reduce_mean(gate) for gate in cdrp_variables.gate_variables.values()]) / len(
        #     cdrp_variables.gate_variables)
        l1_loss = l1_loss * penalty
        total_loss = l1_loss + cross_entropy
        optimizer = tf.train.MomentumOptimizer(
            lr,
            0.9,
            # use_nesterov=False,
            use_nesterov=True,
        )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(
            optimizer.minimize(
                total_loss,
                global_step=global_step,
                var_list=list(cdrp_variables.gate_variables.values()),
            ),
            update_ops,
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            training_hooks=[
                FetchGatesHook(
                    cdrp_variables,
                    {
                        "loss": total_loss,
                        "predictions": softmax,
                        # "l1_loss": l1_loss,
                        # "cross_entropy": cross_entropy,
                        # "gradient": tf.gradients(
                        #     total_loss,
                        #     list(cdrp_variables.gate_variables.values()),
                        # ),
                        # "optimizer_gradient": optimizer.compute_gradients(
                        #     total_loss,
                        #     list(cdrp_variables.gate_variables.values()),
                        # ),
                        # "global_step": global_step,
                    },
                    class_id=class_id,
                )
            ],
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            # predictions=softmax,
            predictions={"logits": logits, "softmax": softmax},
        )


def reconstruct_cdrp_from_tf(
    create_model,
    input_fn,
    model_dir: str,
    class_id: int = None,
    data_format: str = "channels_first",
    parallel: int = 1,
    batch_size: int = 1,
) -> Dict[str, np.ndarray]:
    variables = Variables()
    model_dir = abspath(model_dir)
    model_function = partial(
        model_fn, create_model=create_model, cdrp_variables=variables
    )

    if data_format is None:
        data_format = (
            "channels_first" if tf.test.is_built_with_cuda() else "channels_last"
        )
    estimator_config = tf.estimator.RunConfig(
        session_config=new_session_config(parallel=parallel),
        # save_checkpoints_steps=None,
        # save_checkpoints_secs=None,
        keep_checkpoint_max=1,
    )
    tmp_dir = tempfile.mkdtemp(dir="/dev/shm")
    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=tmp_dir,
        params={
            "data_format": data_format,
            "class_id": class_id,
            "batch_size": batch_size,
        },
        config=estimator_config,
        warm_start_from=WarmStartSettings(
            ckpt_to_initialize_from=model_dir,
            vars_to_warm_start=".*(bias|kernel)|global_step",
        ),
    )

    results = list(classifier.predict(input_fn=input_fn))
    variables.original_logits = np.array([result["logits"] for result in results])
    variables.original_predictions = np.array([result["softmax"] for result in results])

    for epoch in range(30):
        classifier.train(input_fn=input_fn)

    shutil.rmtree(tmp_dir)
    gates = variables.gates if variables.correct else variables.last_gates
    gates = {name: gate[0] for name, gate in gates.items()}
    return gates
    # return {}


def benchmark_cdrp():
    class_id = 1
    image_id = 0
    model_config = ALEXNET_CDRP.with_model_dir("tf/alexnet/model_import")
    # model_config = RESNET_50_CDRP
    # model_config = VGG_16_CDRP
    data_dir = IMAGENET_RAW_DIR
    model_dir = abspath(model_config.model_dir)
    create_model = lambda: model_config.network_class()
    batched_input_fn = lambda: imagenet_raw.test(
        data_dir,
        class_id,
        image_id,
        class_from_zero=model_config.class_from_zero,
        preprocessing_fn=model_config.preprocessing_fn,
        batch=1,
    )

    # create_original_model = lambda: model_config.network_class(with_gates=False)
    # input_fn = lambda: imagenet_raw.test(data_dir, class_id, image_id,
    #                                      class_from_zero=model_config.class_from_zero,
    #                                      preprocessing_fn=model_config.preprocessing_fn)
    # predicted_label = predict(
    #     create_model=create_original_model,
    #     input_fn=input_fn,
    #     model_dir=model_dir,
    # )
    #
    # if predicted_label != class_id:
    #     return None

    reconstruct_cdrp_from_tf(
        create_model=create_model,
        class_id=class_id,
        input_fn=batched_input_fn,
        model_dir=model_dir,
        batch_size=1,
        # parallel=40,
    )


if __name__ == "__main__":
    benchmark_cdrp()
