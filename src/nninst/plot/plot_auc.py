from functools import partial
from typing import List

from matplotlib.ticker import MaxNLocator
from sklearn import svm
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import Lasso, SGDClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle

from nninst import mode
from nninst.backend.tensorflow.attack.cdrp import (
    alexnet_imagenet_example_cdrp,
    resnet_18_cifar100_example_cdrp,
    vgg_16_imagenet_example_cdrp,
)
from nninst.backend.tensorflow.attack.common import (
    alexnet_imagenet_example_stat,
    resnet_50_imagenet_example_stat,
)
from nninst.backend.tensorflow.model import VGG16, AlexNet, ResNet50
from nninst.backend.tensorflow.model.config import ALEXNET, RESNET_50
from nninst.backend.tensorflow.model.densenet import DenseNet
from nninst.backend.tensorflow.model.resnet_18_cifar10 import ResNet18Cifar10
from nninst.backend.tensorflow.model.resnet_18_cifar100 import ResNet18Cifar100
from nninst.op import *
from nninst.plot.prelude import *
from nninst.trace import TraceKey, density_name
from nninst.utils.ray import ray_init, ray_map

if __name__ == "__main__":
    mode.local()
    # mode.debug()
    ray_init()
    # threshold = 0.1
    # threshold = 0.3
    threshold = 0.5
    # threshold = 0.7
    # threshold = 0.9
    # threshold = 1.0
    absolute_threshold = None
    # absolute_threshold = 0.05
    # absolute_threshold = 0.1
    # absolute_threshold = 0.2
    # absolute_threshold = 0.3
    # absolute_threshold = 0.4
    normal_example_label = -1
    adversarial_example_label = -normal_example_label
    summary_path_template = (
        lambda threshold, attack_name, metric_type, model, label: f"{model}_{metric_type}_metrics_per_layer_{threshold:.1f}_{attack_name}{to_suffix(label)}.csv"
    )
    save_path_template = (
        lambda model, label, comparison_type, classifier: f"{model}_roc_v2[{label}][comparison={comparison_type}][classifier={classifier}].pdf"
    )

    classifier_configs = {
        "linear": Config(
            clf=SGDClassifier(
                loss="hinge",
                penalty="elasticnet",
                max_iter=10000,
                tol=1e-4,
                fit_intercept=False,
                l1_ratio=0.5,
            ),
            #         clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=10000, tol=1e-4, fit_intercept=False),
            #         clf = SGDClassifier(loss="hinge", penalty="l1", max_iter=10000, tol=1e-4, fit_intercept=False),
            is_linear=True,
            is_one_class=False,
            clf_name="Linear",
        ),
        "svm": Config(clf=svm.SVC()),
        "linear_svm": Config(
            clf=svm.LinearSVC(loss="squared_hinge", penalty="l1", dual=False),
            is_linear=True,
            is_one_class=False,
        ),
        "lasso": Config(clf=Lasso(positive=True), is_linear=True, is_one_class=False),
        "rf": Config(
            clf=RandomForestClassifier(n_estimators=100, max_depth=None),
            is_linear=False,
            is_one_class=False,
            clf_name="Random forest",
        ),
        "ada": Config(
            clf=AdaBoostClassifier(),
            is_linear=False,
            is_one_class=False,
            clf_name="Adaboost",
        ),
        "gb": Config(
            clf=GradientBoostingClassifier(),
            is_linear=False,
            is_one_class=False,
            clf_name="Gradient boosting",
        ),
        "if": Config(
            clf=IsolationForest(n_estimators=100),
            is_linear=False,
            is_one_class=True,
            clf_name="Isolation forest",
        ),
        "ocsvm": Config(
            clf=OneClassSVM(),
            is_linear=False,
            is_one_class=True,
            clf_name="One class SVM",
        ),
    }
    metric_configs = {
        "diff": Config(
            metric_name="Confidence",
            metric_key="overlap_ratio_diff",
            metric=lambda top1_overlap_size, top1_trace_size, top2_overlap_size, top2_trace_size: (
                top1_overlap_size / top1_trace_size
            )
            - (top2_overlap_size / top2_trace_size),
        ),
        "ratio": Config(
            metric_name="Rank1",
            metric_key="overlap_ratio",
            metric=lambda top1_overlap_size, top1_trace_size, top2_overlap_size, top2_trace_size: top1_overlap_size
            / top1_trace_size,
        ),
        "rank2": Config(
            metric_name="Rank2",
            metric_key="overlap_ratio",
            metric=lambda top1_overlap_size, top1_trace_size, top2_overlap_size, top2_trace_size: -top2_overlap_size
            / top2_trace_size,
        ),
    }
    metric_type_configs = {
        "ideal": Config(metric_type="ideal"),
        "real": Config(metric_type="real"),
    }
    model_configs = {
        "alexnet": Config(
            model_name="alexnet_imagenet",
            model=AlexNet,
            label="import",
            # label="without_dropout",
            cdrp_fn=alexnet_imagenet_example_cdrp,
            stat_fn=alexnet_imagenet_example_stat,
            gates={
                1: "conv2d/gate1:0",
                2: "conv2d_1/gate2:0",
                3: "conv2d_2/gate3:0",
                4: "conv2d_3/gate4:0",
                5: "conv2d_4/gate5:0",
            },
            class_ids=range(1000),
            image_ids=range(1),
            labelrotation=60,
            figsize=(8, 5),
            x_border=[4.5],
        ),
        "resnet_18_cifar100": Config(
            model_name="resnet_18_cifar100",
            model=ResNet18Cifar100,
            label=None,
            cdrp_fn=resnet_18_cifar100_example_cdrp,
            gates={
                1: "gate1:0",
                2: "gate2:0",
                3: "gate3:0",
                4: "gate4:0",
                5: "gate5:0",
                6: "gate6:0",
                7: "gate7:0",
                8: "gate8:0",
            },
            class_ids=range(100),
            image_ids=range(10),
            labelrotation=60,
            figsize=(8, 5),
            x_border=[4.5],
        ),
        "resnet_18_cifar10": Config(
            model_name="resnet_18_cifar10",
            model=ResNet18Cifar10,
            label=None,
            labelrotation=60,
            figsize=(8, 5),
            x_border=[4.5],
        ),
        "densenet_cifar10": Config(
            model_name="densenet_cifar10",
            model=DenseNet,
            label=None,
            labelrotation=60,
            figsize=(8, 5),
            x_border=[4.5],
        ),
        "alexnet_per_channel": Config(
            model_name="alexnet_imagenet",
            model=AlexNet,
            label="import_per_channel",
            labelrotation=60,
            figsize=(8, 5),
            x_border=[4.5],
        ),
        "alexnet_weight": Config(
            model_name="alexnet_imagenet",
            model=AlexNet,
            label="import_weight",
            labelrotation=60,
            figsize=(8, 5),
            x_border=[4.5],
        ),
        "resnet_50": Config(
            model_name="resnet_50_imagenet",
            model=ResNet50,
            label=None,
            stat_fn=resnet_50_imagenet_example_stat,
            labelrotation=90,
            figsize=(8, 5),
            x_border=[0.5, 9.5, 21.5, 39.5, 48.5],
        ),
        "vgg_16": Config(
            model_name="vgg_16_imagenet",
            model=VGG16,
            label=None,
            cdrp_fn=vgg_16_imagenet_example_cdrp,
            gates=[
                "vgg_16/conv1/gate1:0",
                "vgg_16/conv1/gate2:0",
                "vgg_16/conv2/gate3:0",
                "vgg_16/conv2/gate4:0",
                "vgg_16/conv3/gate5:0",
                "vgg_16/conv3/gate6:0",
                "vgg_16/conv3/gate7:0",
                "vgg_16/conv4/gate8:0",
                "vgg_16/conv4/gate9:0",
                "vgg_16/conv4/gate10:0",
                "vgg_16/conv5/gate11:0",
                "vgg_16/conv5/gate12:0",
                "vgg_16/conv5/gate13:0",
                "vgg_16/gate14:0",
                "vgg_16/gate15:0",
            ],
            labelrotation=90,
            figsize=(8, 5),
            x_border=[],
        ),
    }
    benchmark_configs = {
        "basic": Config(attack_names=["BIM", "JSMA", "DeepFool", "CWL2", "FGSM"]),
        "targeted": Config(attack_names=["CWL2", "CWL2_target=500"]),
        "universal": Config(
            attack_names=[
                "patch_scale=0.1",
                "patch_scale=0.2",
                "patch_scale=0.3",
                "patch_scale=0.4",
                "patch_scale=0.5",
            ]
        ),
        "new_universal": Config(
            attack_names=[
                "new_patch_scale=0.1",
                "new_patch_scale=0.2",
                "new_patch_scale=0.3",
                "new_patch_scale=0.4",
                "new_patch_scale=0.5",
            ]
        ),
        "negative": Config(attack_names=["CWL2", "Random"]),
        "confidence": Config(
            attack_names=[
                "CWL2",
                "CWL2_confidence=3.5",
                "CWL2_confidence=14",
                "CWL2_confidence=28",
            ]
        ),
        "extended": Config(
            attack_names=[
                "BIM",
                "JSMA",
                "DeepFool",
                "CWL2",
                "FGSM",
                "CWL2_target=500",
                "FGSM_iterative_targeted",
                "patch_scale=0.3",
                "Random",
            ]
        ),
        "vgg": Config(
            attack_names=[
                "FGSM",
                # "FGSM_iterative_targeted",
            ]
        ),
        "test": Config(
            attack_names=[
                "DeepFool",
                "FGSM",
                # "FGSM_targeted",
                # "FGSM_iterative_targeted",
                "BIM",
                "JSMA",
                "CWL2",
                # "Adaptive_layer1",
                # "Adaptive_layer2",
                # "Adaptive_layer3",
                # "Adaptive_layer4",
                # "Adaptive_layer5",
                # "Adaptive_layer6",
                # "Adaptive_layer7",
                # "Adaptive_layer8",
                # "Adaptive_layer9",
                # "Adaptive_cos_layer9",
                # "Adaptive_return_late",
                # "Adaptive_random_start",
                # "Adaptive_iterations_400",
                # "Adaptive_layer4_iterations_400",
                # "CWL2_confidence=3.5",
                # "CWL2_confidence=14",
                # "CWL2_confidence=28",
                # "CWL2_target=500",
                # "CWL2_confidence=28_target=500",
                # "patch",
                # "patch_scale=0.1",
                # "patch_scale=0.2",
                # "patch_scale=0.3",
                # "patch_scale=0.4",
                # "patch_scale=0.5",
                # "negative_example",
                # "negative_example_top5",
                # "negative_example_out_of_top5",
                # "Random",
            ]
        ),
    }

    def shorten(layer: str) -> str:
        return layer[: layer.find("/")]

    attack_to_label = {
        "CWL2_confidence=3.5": "confidence=3.5",
        "CWL2_confidence=14": "confidence=14",
        "CWL2_confidence=28": "confidence=28",
        "CWL2_target=500": "CWL2_targeted",
        "FGSM_iterative_targeted": "FGSM_targeted",
        "patch_scale=0.1": "scale=0.1",
        "patch_scale=0.2": "scale=0.2",
        "patch_scale=0.3": "Patch",
        "patch_scale=0.4": "scale=0.4",
        "patch_scale=0.5": "scale=0.5",
    }

    def get_metric(summary, example_type: str, layer: str, metric_key: str):
        if example_type == "original":
            top1_prefix = f"{example_type}.origin"
            top2_prefix = f"{example_type}.target"
        elif example_type == "adversarial":
            top1_prefix = f"{example_type}.target"
            top2_prefix = f"{example_type}.origin"
        top1_overlap_size = summary[f"{top1_prefix}.{layer}.overlap_size_in_class"]
        top2_overlap_size = summary[f"{top2_prefix}.{layer}.overlap_size_in_class"]
        top1_trace_size = summary[f"{top1_prefix}.{layer}.overlap_size_total"]
        top2_trace_size = summary[f"{top2_prefix}.{layer}.overlap_size_total"]
        return metric_configs[metric_key].metric(
            top1_overlap_size, top1_trace_size, top2_overlap_size, top2_trace_size
        )

    def get_metric_from_rank(
        summary_file_fn, attack_name: str, layer_names: str, rank_num: int
    ):
        summary = pd.read_csv(summary_file_fn(attack_name=attack_name, rank=1))
        rank = 2
        while rank_num >= rank:
            summary = summary.merge(
                pd.read_csv(summary_file_fn(attack_name=attack_name, rank=rank)),
                on=["class_id", "image_id"],
                how="inner",
                suffixes=("", f"_rank{rank}"),
            )
            rank += 1
        if attack_name != "normal":
            normal_summary = pd.read_csv(summary_file_fn(attack_name="normal", rank=1))
            summary = summary.merge(
                normal_summary,
                on=["class_id", "image_id"],
                how="inner",
                suffixes=("", "_normal"),
            )
        #     print(attack_name, len(summary))
        metric_list = []
        try:
            for rank in range(1, rank_num + 1):
                for layer in layer_names:
                    if rank == 1:
                        metric_list.append(
                            summary[f"{layer}.overlap_size_in_class"]
                            / summary[f"{layer}.overlap_size_total"]
                        )
                    else:
                        metric_list.append(
                            summary[f"{layer}.overlap_size_in_class_rank{rank}"]
                            / summary[f"{layer}.overlap_size_total_rank{rank}"]
                        )

        except Exception as e:
            raise RuntimeError(
                f"from file: {summary_file_fn(attack_name=attack_name, rank=rank)}"
            ) from e
        #     return metric if rank == 1 else -metric
        return np.stack(metric_list, axis=1)

    available_attacks = {
        # "alexnet": "extended",
        # "alexnet": "basic",
        "alexnet": "test",
        "resnet_18_cifar100": "test",
        "resnet_18_cifar10": "test",
        "densenet_cifar10": "test",
        "alexnet_per_channel": "test",
        "resnet_50": "test",
        "vgg_16": "test",
    }

    extended_patches = [
        "patch_scale=0.1",
        "patch_scale=0.2",
        #     "patch_scale=0.3",
        "patch_scale=0.4",
        "patch_scale=0.5",
    ]

    def get_auc(
        config: Config = None,
        model_key: str = "alexnet",
        test_size: int = 0.9,
        layer_num: int = None,
        attacks_for_train: List[str] = None,
        classifier: str = "linear",
        rank_num: int = 2,
        use_point: bool = False,
        trace_label: str = None,
        variant: str = None,
        compare_with_full: bool = False,
    ):
        graph = config.model.graph().load()
        layers = graph.ops_in_layers(Conv2dOp, DenseOp)
        if layer_num is None:
            used_layers = layers
        else:
            used_layers = layers[(len(layers) - layer_num) :]
        #         used_layers = layers[:layer_num]
        attack_to_train_labels = {}
        attack_to_train_predictions = {}
        attack_to_test_labels = {}
        attack_to_test_predictions = {}

        attack_names = benchmark_configs[available_attacks[model_key]].attack_names
        # for attack_name in config.attack_names:
        if "patch_scale=0.3" in attack_names:
            trained_attack_names = attack_names + extended_patches
            if attacks_for_train is not None:
                attacks_for_train = attacks_for_train + extended_patches
        else:
            trained_attack_names = attack_names
            # trained_attack_names = [
            #     "DeepFool",
            #     "FGSM",
            #     "BIM",
            #     "JSMA",
            #     "CWL2",
            # ]
        random_state = np.random.mtrand._rand
        for attack_name in set(attack_names + trained_attack_names):
            label = config.label
            if variant is not None:
                label = f"{label}_{variant}"
            label = f"{label}_point" if use_point else label
            if compare_with_full:
                label = f"{label}_vs_full"
            if trace_label is not None:
                label = f"{label}_{trace_label}"
            summary_file_fn = lambda attack_name, rank: abspath(
                "metrics/"
                + summary_path_template(
                    threshold=threshold,
                    attack_name=attack_name,
                    metric_type=config.metric_type,
                    model=config.model_name,
                    label=f"{label}_rank{rank}",
                )
            )

            adversarial_metric = get_metric_from_rank(
                summary_file_fn, attack_name, used_layers, rank_num
            )
            normal_metric = get_metric_from_rank(
                summary_file_fn, "normal", used_layers, rank_num
            )
            #         assert adversarial_metric.shape[0] == normal_metric.shape[0]
            predictions = np.concatenate([adversarial_metric, normal_metric])
            row_filter = np.isfinite(predictions).all(axis=1)
            labels = np.concatenate(
                [
                    np.repeat(adversarial_example_label, adversarial_metric.shape[0]),
                    np.repeat(normal_example_label, normal_metric.shape[0]),
                ]
            )
            labels = labels[row_filter]
            predictions = predictions[row_filter]
            labels, predictions = shuffle(
                labels, predictions, random_state=random_state
            )
            (
                train_predictions,
                test_predictions,
                train_labels,
                test_labels,
            ) = train_test_split(
                predictions, labels, test_size=test_size, random_state=random_state
            )
            attack_to_test_labels[attack_name] = test_labels
            attack_to_test_predictions[attack_name] = test_predictions
            attack_to_train_labels[attack_name] = train_labels
            attack_to_train_predictions[attack_name] = train_predictions

        train_labels = np.concatenate(
            [
                attack_to_train_labels[attack_name]
                for attack_name in (attacks_for_train or trained_attack_names)
            ]
        )
        train_predictions = np.concatenate(
            [
                attack_to_train_predictions[attack_name]
                for attack_name in (attacks_for_train or trained_attack_names)
            ]
        )
        train_labels, train_predictions = shuffle(train_labels, train_predictions)

        clf = classifier_configs[classifier].clf
        if classifier_configs[classifier].is_one_class:
            clf.fit(train_predictions[train_labels == normal_example_label])
        else:
            clf.fit(train_predictions, train_labels)
        #     if classifier_configs[classifier].is_linear:
        #         clf.coef_[clf.coef_ < 0] = 0

        path = ensure_dir(abspath("store/plot_auc.pkl"))
        with (open(path, "wb")) as file:
            pickle.dump(clf, file)
        # print(np.mean([estimator.get_depth() for estimator in clf.estimators_]))
        attack_to_auc = {}
        for attack_name in attack_names:
            labels = attack_to_test_labels[attack_name]
            predictions = attack_to_test_predictions[attack_name]
            if classifier_configs[classifier].is_linear:
                y_score = clf.decision_function(predictions)
            elif classifier_configs[classifier].is_one_class:
                y_score = clf.decision_function(predictions) * normal_example_label
            else:
                # start_time = time.time()
                y_score = clf.predict_proba(predictions)[:, 1]
                # used_time = time.time() - start_time
                # print(f"use: {used_time}s for {len(labels)} samples, avg: {used_time/len(labels)}s")
            fpr, tpr, thresholds = metrics.roc_curve(
                labels, y_score, drop_intermediate=False
            )
            # fix_fpr = True
            # # fix_fpr = False
            # if fix_fpr:
            #     fpr_indices = np.nonzero(fpr <= 0.146)  # ResNet-50
            #     # fpr_indices = np.nonzero(fpr <= 0.038)  # DenseNet
            #     fpr_index = fpr_indices[0][-1]
            #     print(f"tpr: {tpr[fpr_index]:.4f}, fpr: {fpr[fpr_index]:.4f}")
            # else:
            #     tpr_indices = np.nonzero(tpr >= 0.92)  # ResNet-50
            #     # tpr_indices = np.nonzero(tpr >= 0.93)  # DenseNet
            #     tpr_index = tpr_indices[0][0]
            #     print(f"fpr: {fpr[tpr_index]:.4f}, tpr: {tpr[tpr_index]:.4f}")
            roc_auc = metrics.auc(fpr, tpr)
            attack_to_auc[attack_name] = roc_auc
        return attack_to_auc, train_labels.size

    def get_cdrp(cdrp_fn, gates, attack_name: str, class_id: int, image_id: int):
        cdrp = cdrp_fn(
            attack_name=attack_name,
            attack_fn=None,
            generate_adversarial_fn=None,
            class_id=class_id,
            image_id=image_id,
        ).load()
        if cdrp is not None:
            cdrp = np.concatenate([cdrp[gate].flatten() for gate in gates.values()])
        return cdrp

    def get_auc_cdrp(
        config: Config = None,
        model_key: str = "alexnet",
        test_size: int = 0.9,
        layer_num: int = None,
        attacks_for_train: List[str] = None,
        classifier: str = "linear",
    ):
        graph = config.model.graph().load()
        layers = graph.ops_in_layers(Conv2dOp, DenseOp)
        if layer_num is None:
            gates = config.gates
        else:
            gates = {
                layer_id: config.gates[layer_id]
                for layer_id in range(len(layers) - layer_num, len(layers))
                if layer_id in config.gates
            }
        #         gates = {layer_id: config.gates[layer_id] for layer_id in range(layer_num) if layer_id in config.gates}
        if len(gates) == 0:
            return {}, 0

        attack_to_train_labels = {}
        attack_to_train_predictions = {}
        attack_to_test_labels = {}
        attack_to_test_predictions = {}

        attack_names = benchmark_configs[available_attacks[model_key]].attack_names
        # for attack_name in config.attack_names:
        if "patch_scale=0.3" in attack_names:
            trained_attack_names = attack_names + extended_patches
            if attacks_for_train is not None:
                attacks_for_train = attacks_for_train + extended_patches
        else:
            trained_attack_names = attack_names
        for attack_name in trained_attack_names:
            negative_cdrps = np.stack(
                list(
                    filter_not_null(
                        [
                            get_cdrp(
                                cdrp_fn=config.cdrp_fn,
                                gates=gates,
                                attack_name=attack_name,
                                class_id=class_id,
                                image_id=image_id,
                            )
                            for class_id in config.class_ids
                            for image_id in config.image_ids
                        ]
                    )
                ),
                axis=0,
            )
            negative_labels = np.repeat(
                adversarial_example_label, negative_cdrps.shape[0]
            )
            positive_cdrps = np.stack(
                list(
                    filter_not_null(
                        [
                            get_cdrp(
                                cdrp_fn=config.cdrp_fn,
                                gates=gates,
                                attack_name="original",
                                class_id=class_id,
                                image_id=image_id,
                            )
                            for class_id in config.class_ids
                            for image_id in config.image_ids
                        ]
                    )
                ),
                axis=0,
            )
            positive_labels = np.repeat(normal_example_label, positive_cdrps.shape[0])
            #         print(negative_cdrps.shape)
            #         print(positive_cdrps.shape)
            cdrps = np.concatenate([negative_cdrps, positive_cdrps])
            #         print(cdrps.shape)
            labels = np.concatenate([negative_labels, positive_labels])
            filter_dense = np.count_nonzero(cdrps, axis=1) != cdrps.shape[1]
            cdrps = cdrps[filter_dense]
            labels = labels[filter_dense]
            predictions = cdrps
            labels, predictions = shuffle(labels, predictions)
            (
                train_predictions,
                test_predictions,
                train_labels,
                test_labels,
            ) = train_test_split(predictions, labels, test_size=test_size)
            attack_to_test_labels[attack_name] = test_labels
            attack_to_test_predictions[attack_name] = test_predictions
            attack_to_train_labels[attack_name] = train_labels
            attack_to_train_predictions[attack_name] = train_predictions

        train_labels = np.concatenate(
            [
                attack_to_train_labels[attack_name]
                for attack_name in (attacks_for_train or trained_attack_names)
            ]
        )
        train_predictions = np.concatenate(
            [
                attack_to_train_predictions[attack_name]
                for attack_name in (attacks_for_train or trained_attack_names)
            ]
        )
        train_labels, train_predictions = shuffle(train_labels, train_predictions)

        #     scaler = StandardScaler()
        #     scaler.fit(train_predictions)  # Don't cheat - fit only on training data
        #     train_predictions = scaler.transform(train_predictions)

        #     print(train_predictions.shape)
        clf = classifier_configs[classifier].clf
        clf.fit(train_predictions, train_labels)
        #     if classifier_configs[classifier].is_linear:
        #         clf.coef_[clf.coef_ < 0] = 0

        attack_to_auc = {}
        for attack_name in attack_names:
            labels = attack_to_test_labels[attack_name]
            predictions = attack_to_test_predictions[attack_name]
            #         predictions = scaler.transform(predictions)
            if classifier_configs[classifier].is_linear:
                y_score = clf.decision_function(predictions)
            else:
                y_score = clf.predict_proba(predictions)[:, 1]
            fpr, tpr, thresholds = metrics.roc_curve(labels, y_score)
            roc_auc = metrics.auc(fpr, tpr)
            attack_to_auc[attack_name] = roc_auc
        return attack_to_auc, train_labels.size

    def get_stat(stat_fn, layers, attack_name: str, class_id: int, image_id: int):
        stat = stat_fn(
            attack_name=attack_name,
            attack_fn=None,
            generate_adversarial_fn=None,
            class_id=class_id,
            image_id=image_id,
        ).load()
        if stat is not None:
            stat = np.concatenate([stat[layer] for layer in layers])
        return stat

    def get_auc_stat(
        stat_name: str,
        config: Config = None,
        model_key: str = "alexnet",
        test_size: int = 0.8,
        layer_num: int = None,
        attacks_for_train: List[str] = None,
        classifier: str = "linear",
    ):
        graph = config.model.graph().load()
        layers = graph.ops_in_layers(Conv2dOp, DenseOp)[:-1]
        if layer_num is None:
            used_layers = layers
        else:
            #         if layer_num == 1:
            #             layer_num = 2
            #         used_layers = layers[(len(layers) - layer_num):]
            used_layers = layers[:layer_num]

        attack_to_train_labels = {}
        attack_to_train_predictions = {}
        attack_to_test_labels = {}
        attack_to_test_predictions = {}

        attack_names = benchmark_configs[available_attacks[model_key]].attack_names
        # for attack_name in config.attack_names:
        if "patch_scale=0.3" in attack_names:
            trained_attack_names = attack_names + extended_patches
            if attacks_for_train is not None:
                attacks_for_train = attacks_for_train + extended_patches
        else:
            trained_attack_names = attack_names
        for attack_name in trained_attack_names:
            #         negative_stats = np.stack(filter_not_null([get_stat(stat_fn=partial(config.stat_fn, stat_name=stat_name), layers=used_layers, attack_name=attack_name, class_id=class_id, image_id=0) for class_id in range(1000)]), axis=0)
            #         negative_stats = np.stack(filter_not_null([get_stat(stat_fn=partial(config.stat_fn, stat_name=stat_name), layers=used_layers, attack_name="original", class_id=class_id, image_id=0) for class_id in range(1, 1000)]), axis=0)
            negative_stats = np.stack(
                list(
                    filter_not_null(
                        [
                            get_stat(
                                stat_fn=partial(config.stat_fn, stat_name=stat_name),
                                layers=used_layers,
                                attack_name="original",
                                class_id=class_id,
                                image_id=0,
                            )
                            for class_id in range(2, 1001)
                        ]
                    )
                ),
                axis=0,
            )
            negative_labels = np.repeat(
                adversarial_example_label, negative_stats.shape[0]
            )
            #         positive_stats = np.stack(filter_not_null([get_stat(stat_fn=partial(config.stat_fn, stat_name=stat_name), layers=used_layers, attack_name="original", class_id=class_id, image_id=0) for class_id in range(1000)]), axis=0)
            #         positive_stats = np.stack(filter_not_null([get_stat(stat_fn=partial(config.stat_fn, stat_name=stat_name), layers=used_layers, attack_name="original", class_id=0, image_id=image_id) for image_id in range(1000)]), axis=0)
            positive_stats = np.stack(
                list(
                    filter_not_null(
                        [
                            get_stat(
                                stat_fn=partial(config.stat_fn, stat_name=stat_name),
                                layers=used_layers,
                                attack_name="original",
                                class_id=1,
                                image_id=image_id,
                            )
                            for image_id in range(1000)
                        ]
                    )
                ),
                axis=0,
            )
            positive_labels = np.repeat(normal_example_label, positive_stats.shape[0])
            #         print(negative_cdrps.shape)
            #         print(positive_cdrps.shape)
            stats = np.concatenate([negative_stats, positive_stats])
            #         print(cdrps.shape)
            labels = np.concatenate([negative_labels, positive_labels])
            predictions = stats
            labels, predictions = shuffle(labels, predictions)
            (
                train_predictions,
                test_predictions,
                train_labels,
                test_labels,
            ) = train_test_split(predictions, labels, test_size=test_size)
            attack_to_test_labels[attack_name] = test_labels
            attack_to_test_predictions[attack_name] = test_predictions
            attack_to_train_labels[attack_name] = train_labels
            attack_to_train_predictions[attack_name] = train_predictions

        train_labels = np.concatenate(
            [
                attack_to_train_labels[attack_name]
                for attack_name in (attacks_for_train or trained_attack_names)
            ]
        )
        train_predictions = np.concatenate(
            [
                attack_to_train_predictions[attack_name]
                for attack_name in (attacks_for_train or trained_attack_names)
            ]
        )
        train_labels, train_predictions = shuffle(train_labels, train_predictions)

        scaler = StandardScaler()
        scaler.fit(train_predictions)  # Don't cheat - fit only on training data
        train_predictions = scaler.transform(train_predictions)

        #     print(train_predictions.shape)
        clf = classifier_configs[classifier].clf
        clf.fit(train_predictions, train_labels)
        #     if classifier_configs[classifier].is_linear:
        #         clf.coef_[clf.coef_ < 0] = 0

        attack_to_auc = {}
        for attack_name in attack_names:
            labels = attack_to_test_labels[attack_name]
            predictions = attack_to_test_predictions[attack_name]
            predictions = scaler.transform(predictions)
            if classifier_configs[classifier].is_linear:
                y_score = clf.decision_function(predictions)
            else:
                y_score = clf.predict_proba(predictions)[:, 1]
            fpr, tpr, thresholds = metrics.roc_curve(labels, y_score)
            roc_auc = metrics.auc(fpr, tpr)
            attack_to_auc[attack_name] = roc_auc
        return attack_to_auc, train_labels.size

    # classifiers = ["linear", "rf", "ada", "gb"]
    # classifiers = ["rf", "ada", "gb"]
    # classifiers = ["linear", "rf"]
    classifiers = ["rf"]

    comparison_type_configs = {
        "layer_num": Config(
            xlabel="Layer number",
            comparison_fn=lambda get_auc_fn, value, **kwargs: get_auc_fn(
                layer_num=value, **kwargs
            ),
            comparison_iter=lambda layer_num, *args: list(range(1, layer_num + 1)),
            map_fn=lambda value, **kwargs: value,
        ),
        "attack_num": Config(
            xlabel="Attack number for training",
            comparison_fn=lambda get_auc_fn, value, **kwargs: get_auc_fn(
                attacks_for_train=value, **kwargs
            ),
            comparison_iter=lambda _, attack_names, *args: [
                attack_names[:attack_num]
                for attack_num in range(1, len(attack_names) + 1)
            ],
            map_fn=lambda value, **kwargs: len(value),
        ),
        "train_num": Config(
            xlabel="Training size",
            comparison_fn=lambda get_auc_fn, value, **kwargs: get_auc_fn(
                test_size=value, **kwargs
            ),
            comparison_iter=lambda *args: -np.linspace(0.01, 0.4, 10) + 1,
            map_fn=lambda value, train_num, **kwargs: train_num,
        ),
        "classifier": Config(
            xlabel="Attack",
            comparison_fn=lambda get_auc_fn, value, **kwargs: get_auc_fn(
                classifier=value, **kwargs
            ),
            comparison_iter=lambda *args: classifiers,
            map_fn=lambda value, **kwargs: classifier_configs[value].clf_name,
            #         figsize=(16,6)
        ),
        "rank_num": Config(
            xlabel="Rank",
            comparison_fn=lambda get_auc_fn, value, **kwargs: get_auc_fn(
                rank_num=value, **kwargs
            ),
            comparison_iter=lambda *args: list(range(1, 11)),
            map_fn=lambda value, **kwargs: value,
        ),
    }

    # for AlexNet
    # type_codes = [
    #     # "21111111", # == type2
    #     "21111112",
    #     "21111122",
    #     "21111222",
    #     "21112222",
    #     "21122222",
    #     "21222222",
    #     "22222222",
    #     # "42222222", # == type4
    # ]
    # for ResNet-18
    type_codes = [
        # "211111111111111111", # == type2
        "211111111111111112",
        "211111111111111122",
        "211111111111111222",
        "211111111111112222",
        "211111111111122222",
        "211111111111222222",
        "211111111112222222",
        "211111111122222222",
        "211111111222222222",
        "211111112222222222",
        "211111122222222222",
        "211111222222222222",
        "211112222222222222",
        "211122222222222222",
        "211222222222222222",
        "212222222222222222",
        "222222222222222222",
        # "422222222222222222", # == type4
    ]

    hybrid_backward_trace_configs = {
        f"type{type_code}_trace_from_{threshold:.1f}": Config(
            auc_fn=partial(
                get_auc, trace_label=f"type{type_code}_density_from_{threshold:.1f}"
            ),
            trace_name=f"Type {type_code}",
        )
        for type_code in type_codes
    }
    trace_type_configs = {
        "trace": Config(
            auc_fn=get_auc,
            # trace_name="Str",
            trace_name="Type 1",
        ),
        "point_trace": Config(
            auc_fn=partial(get_auc, use_point=True), trace_name="StrP"
        ),
        # "trace": Config(auc_fn=get_auc, trace_name="Ours"),
        # "point_trace": Config(
        #     auc_fn=partial(get_auc, use_point=True), trace_name="OursP"
        # ),
        "cdrp": Config(auc_fn=get_auc_cdrp, trace_name="CDRPs"),
        "stat_avg": Config(
            auc_fn=partial(get_auc_stat, stat_name="avg"), trace_name="AvgPool"
        ),
        "stat_max": Config(
            auc_fn=partial(get_auc_stat, stat_name="max"), trace_name="MaxPool"
        ),
        "unstr_point_trace_0.01": Config(
            auc_fn=partial(
                get_auc, use_point=True, trace_label="unstructured_density_0.01"
            ),
            trace_name="UnstrP(density=0.01)",
        ),
        "unstr_trace_0.01": Config(
            auc_fn=partial(get_auc, trace_label="unstructured_density_0.01"),
            trace_name="Unstr(density=0.01)",
        ),
        "unstr_trace_0.02": Config(
            auc_fn=partial(get_auc, trace_label="unstructured_density_0.02"),
            trace_name="Unstr(density=0.02)",
        ),
        "unstr_trace_0.05": Config(
            auc_fn=partial(get_auc, trace_label="unstructured_density_0.05"),
            trace_name="Unstr(density=0.05)",
        ),
        "unstr_trace_0.1": Config(
            auc_fn=partial(get_auc, trace_label="unstructured_density_0.1"),
            trace_name="Unstr(density=0.1)",
        ),
        "unstr_trace_0.2": Config(
            auc_fn=partial(get_auc, trace_label="unstructured_density_0.2"),
            trace_name="Unstr(density=0.2)",
        ),
        "unstr_trace_0.005": Config(
            auc_fn=partial(get_auc, trace_label="unstructured_density_0.005"),
            trace_name="Unstr(density=0.005)",
        ),
        "unstr_point_trace_0.001": Config(
            auc_fn=partial(
                get_auc, use_point=True, trace_label="unstructured_density_0.001"
            ),
            trace_name="UnstrP(density=0.001)",
        ),
        "unstr_trace_0.001": Config(
            auc_fn=partial(get_auc, trace_label="unstructured_density_0.001"),
            trace_name="Unstr(density=0.001)",
        ),
        f"unstr_point_trace_from_{threshold:.1f}": Config(
            auc_fn=partial(
                get_auc,
                use_point=True,
                trace_label=f"unstructured_density_from_{threshold:.1f}",
            ),
            trace_name=f"UnstrP(from={threshold:.1f})",
        ),
        f"type2_trace_from_{threshold:.1f}": Config(
            auc_fn=partial(get_auc, trace_label=f"type2_density_from_{threshold:.1f}"),
            trace_name="Type 2",
        ),
        f"trace_[early_stop=12]": Config(
            auc_fn=partial(get_auc, variant="[early_stop=12]"),
            trace_name="Type 1 early",
        ),
        f"type2_trace_from_{threshold:.1f}_[early_stop=10]": Config(
            auc_fn=partial(
                get_auc,
                trace_label=f"type2_density_from_{threshold:.1f}",
                variant="[early_stop=10]",
            ),
            trace_name="Type 2",
        ),
        f"type2_trace_from_{threshold:.1f}_[early_stop=12]": Config(
            auc_fn=partial(
                get_auc,
                trace_label=f"type2_density_from_{threshold:.1f}",
                variant="[early_stop=12]",
            ),
            trace_name="Type 2",
        ),
        f"type4_trace_from_{threshold:.1f}_[early_stop=12]": Config(
            auc_fn=partial(
                get_auc,
                trace_label=f"type4_density_from_{threshold:.1f}",
                variant="[early_stop=12]",
            ),
            trace_name="Type 4",
        ),
        f"type3_trace_from_{threshold:.1f}": Config(
            auc_fn=partial(get_auc, trace_label=f"type3_density_from_{threshold:.1f}"),
            trace_name="Type 3",
        ),
        f"type4_trace_from_{threshold:.1f}": Config(
            auc_fn=partial(get_auc, trace_label=f"type4_density_from_{threshold:.1f}"),
            trace_name="Type 4",
        ),
        f"unstr_trace_from_{threshold:.1f}": Config(
            auc_fn=partial(
                get_auc, trace_label=f"unstructured_density_from_{threshold:.1f}"
            ),
            # trace_name=f"Unstr(from={threshold:.1f})",
            trace_name="Type 5",
        ),
        f"unstr_trace_from_{threshold:.1f}_with_full": Config(
            auc_fn=partial(
                get_auc,
                trace_label=f"unstructured_density_from_{threshold:.1f}",
                compare_with_full=True,
            ),
            trace_name=f"Unstr(from={threshold:.1f}, vs full)",
        ),
        f"per_receptive_field_unstr_trace_from_{threshold:.1f}_with_full": Config(
            auc_fn=partial(
                get_auc,
                trace_label=f"per_receptive_field_unstructured_density_from_{threshold:.1f}",
                compare_with_full=True,
            ),
            trace_name=f"UnstrRF(from={threshold:.1f}, vs full)",
        ),
        f"per_input_unstr_trace_from_{threshold:.1f}_with_full": Config(
            auc_fn=partial(
                get_auc,
                trace_label=f"per_input_unstructured_density_from_{threshold:.1f}",
                compare_with_full=True,
            ),
            trace_name=f"UnstrWI(from={threshold:.1f}, vs full)",
        ),
        f"per_receptive_field_unstr_trace_from_{threshold:.1f}": Config(
            auc_fn=partial(
                get_auc,
                trace_label=f"per_receptive_field_unstructured_density_from_{threshold:.1f}",
            ),
            # trace_name=f"UnstrRF(from={threshold:.1f})",
            trace_name="Type 6",
        ),
        f"type7_trace_from_{threshold:.1f}": Config(
            auc_fn=partial(get_auc, trace_label=f"type7_density_from_{threshold:.1f}"),
            trace_name="Type 7",
        ),
        f"per_input_unstr_trace_from_{threshold:.1f}": Config(
            auc_fn=partial(
                get_auc,
                trace_label=f"per_input_unstructured_density_from_{threshold:.1f}",
            ),
            # trace_name=f"UnstrWI(from={threshold:.1f})",
            trace_name="Type 8",
        ),
        **hybrid_backward_trace_configs,
    }
    if absolute_threshold is not None:
        trace_type_configs.update(
            {
                f"type4_trace_from_{threshold:.1f}_absolute_{absolute_threshold:.2f}": Config(
                    auc_fn=partial(
                        get_auc,
                        trace_label=f"type4_density_from_{threshold:.1f}_absolute_{absolute_threshold:.2f}",
                    ),
                    trace_name="Type 4 ABS",
                ),
            }
        )

    def plot_comparison(
        model_key: str,
        comparison_type_key: str,
        classifier: str,
        trace_types: List[str],
        metric_type: str,
        repeated_times: int = 5,
        cache: bool = False,
    ):
        config = (
            metric_type_configs[metric_type]
            + model_configs[model_key]
            + comparison_type_configs[comparison_type_key]
        )
        aucs = []
        attack_names = benchmark_configs[available_attacks[model_key]].attack_names
        total_attack_num = len(attack_names)
        graph = config.model.graph().load()
        layers = graph.ops_in_layers(Conv2dOp, DenseOp)
        layer_num = len(layers)
        # layer_num = 10
        fig = plt.figure(figsize=config.figsize)
        ax = plt.axes()

        def get_auc_fn(iter_value, i, trace_type):
            new_config = config + trace_type_configs[trace_type]
            if comparison_type_key == "classifier":
                attack_to_auc, train_num = new_config.comparison_fn(
                    new_config.auc_fn,
                    iter_value,
                    config=new_config,
                    model_key=model_key,
                )
            else:
                attack_to_auc, train_num = new_config.comparison_fn(
                    new_config.auc_fn,
                    iter_value,
                    config=new_config,
                    model_key=model_key,
                    classifier=classifier,
                )
            return iter_value, trace_type, attack_to_auc, train_num

        save_path = "plot/" + save_path_template(
            model=model_key,
            label=config.label,
            comparison_type=comparison_type_key,
            classifier=classifier,
        )
        csv_path = abspath(save_path).replace(".pdf", ".csv")
        if cache and os.path.exists(csv_path):
            aucs = pd.read_csv(csv_path)
        else:
            results = ray_map(
                get_auc_fn,
                [
                    (iter_value, i, trace_type)
                    for iter_value, i, trace_type in itertools.product(
                        config.comparison_iter(layer_num, attack_names),
                        range(repeated_times),
                        trace_types,
                    )
                ],
                chunksize=1,
                out_of_order=False,
                num_gpus=0,
            )
            for iter_value, trace_type, attack_to_auc, train_num in results:
                for attack_name, auc in attack_to_auc.items():
                    #                 if attack_name not in ["Random", "patch_scale=0.3"]:
                    if attack_name not in []:
                        aucs.append(
                            {
                                "attack": attack_to_label[attack_name]
                                if attack_name in attack_to_label
                                else attack_name,
                                "auc": auc,
                                "type": trace_type_configs[trace_type].trace_name,
                                comparison_type_key: config.map_fn(
                                    value=iter_value, train_num=train_num
                                ),
                            }
                        )

            #     for iter_value, i in itertools.product(config.comparison_iter(layer_num, attack_names), range(1)):
            #         if comparison_type_key == "classifier":
            #             attack_to_auc, train_num = config.comparison_fn(config.auc_fn, iter_value, config=config, model_key=model_key)
            #         else:
            #             attack_to_auc, train_num = config.comparison_fn(config.auc_fn, iter_value, config=config, model_key=model_key, classifier=classifier)
            #         for attack_name, auc in attack_to_auc.items():
            #             aucs.append({
            #                 "attack": attack_to_label[attack_name] if attack_name in attack_to_label else attack_name,
            #                 "auc": auc,
            #                 comparison_type_key: config.map_fn(value=iter_value, train_num=train_num)
            #             })

            aucs = pd.DataFrame(aucs)
            aucs = (
                aucs.groupby([comparison_type_key, "type", "attack"])
                .mean()
                .reset_index()
            )
            aucs.to_csv(ensure_dir(csv_path), index=False)
        print(save_path)

        # filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
        dash_styles = [
            "",
            (4, 1.5),
            (1, 1),
            (3, 1, 1.5, 1),
            (5, 1, 1, 1),
            (5, 1, 2, 1, 2, 1),
            (2, 2, 3, 1.5),
            (1, 2.5, 3, 1.2),
        ]
        if comparison_type_key == "classifier":
            x_name = "attack"
            ax = sns.lineplot(
                x=x_name,
                y="auc",
                hue=comparison_type_key,
                ci=None,
                data=aucs,
                ax=ax,
                style="type",
                sort=False,
                dashes=dash_styles,
                palette=sns.color_palette("tab10", n_colors=len(classifiers), desat=1),
                #              units="image", estimator=None, lw=0.5,
                #              err_style="bars",
            )
        #         ours = aucs[aucs["type"] == "Ours"]
        #         bar = sns.barplot(x=x_name, y="auc", hue=comparison_type_key, ci=None, data=aucs, ax=ax, palette=sns.color_palette("tab10", n_colors=len(classifiers), desat=1))
        #         for i,thisbar in enumerate(bar.patches):
        #             thisbar.set_hatch("////")
        #         cdrps = aucs[aucs["type"] == "CDRPs"]
        #         bar = sns.barplot(x=x_name, y="auc", hue=comparison_type_key, ci=None, data=aucs, ax=ax, palette=sns.color_palette("tab10", n_colors=len(classifiers), desat=1))

        else:
            x_name = comparison_type_key
            ax = sns.lineplot(
                x=x_name,
                y="auc",
                hue="attack",
                ci=None,
                data=aucs,
                ax=ax,
                markers=True,
                style="type",
                sort=False,
                dashes=dash_styles,
                palette=sns.color_palette("tab10", n_colors=total_attack_num, desat=1),
                #              units="image", estimator=None, lw=0.5,
                #              err_style="bars",
            )
        # ax.set_yscale("log")
        ax.set(xlabel=config.xlabel, ylabel="AUC")
        # ax.legend(bbox_to_anchor=(1, 0), loc="lower right", ncol=1)
        # ax.set_title(f"{config.metric_name} per layer")
        if comparison_type_key == "classifier":
            ax.tick_params(axis="x", labelrotation=45)
            plt.xticks(ha="right")
        ax.set_ylim(top=1)
        ax.set_ylim(bottom=0.5)
        # ax.xaxis.set_ticks(range(1, attack_num + 1))
        # ax.xaxis.set_ticklabels([])
        # ax.xaxis.set_ticklabels(map(shorten, layers))
        #     ax.set_xlim([aucs[x_name].min(), aucs[x_name].max()])
        if comparison_type_key != "classifier":
            ax.xaxis.set_major_locator(
                MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], integer=True)
            )
        handles, labels = ax.get_legend_handles_labels()
        #     print(labels)
        pos = np.where(np.array(labels) == "type")[0][0]
        #     handles = handles[:pos] + handles[pos+1:]
        #     labels = labels[:pos] + labels[pos+1:]
        #     if model_key != "alexnet":
        #         handles = handles[:-1]
        #         labels = labels[:-1]
        handles = handles[:pos] + handles[pos + 1 :]
        labels = labels[:pos] + labels[pos + 1 :]
        if len(trace_types) == 1:
            labels = labels[:-1]
        ax.legend(
            handles[1:],
            labels[1:],
            ncol=1,
            title="",
            bbox_to_anchor=(1, 0.5),
            loc="center left",
        )
        #     ax.legend(handles[2:], labels[2:], ncol=1, title="", bbox_to_anchor=(1, 0.5), loc="center left")
        plt.tight_layout()
        plt.savefig(ensure_dir(abspath(save_path)), bbox_inches="tight")

    # metric_type = "ideal"
    metric_type = "real"
    hybrid_backward_trace_types = [
        f"type{type_code}_trace_from_{threshold:.1f}" for type_code in type_codes
    ]
    trace_types = [
        "trace",  # type1
        # "trace_[early_stop=12]",  # type1
        # "point_trace",
        # "cdrp",
        # "stat_avg",
        # "stat_max",
        # "unstr_trace_0.2",
        # "unstr_trace_0.1",
        # "unstr_trace_0.05",
        # "unstr_trace_0.02",
        # "unstr_point_trace_0.01",
        # "unstr_trace_0.01",
        # "unstr_trace_0.005",
        # "unstr_point_trace_0.001",
        # "unstr_trace_0.001",
        # f"unstr_point_trace_from_{threshold:.1f}",
        # f"type2_trace_from_{threshold:.1f}",
        # f"type2_trace_from_{threshold:.1f}_[early_stop=12]",
        # f"type4_trace_from_{threshold:.1f}_[early_stop=12]",
        # f"type3_trace_from_{threshold:.1f}",
        # *hybrid_backward_trace_types,
        # f"type4_trace_from_{threshold:.1f}",
        # f"type4_trace_from_{threshold:.1f}_absolute_{absolute_threshold:.2f}",
        # f"unstr_trace_from_{threshold:.1f}",  # type5
        # f"unstr_trace_from_{threshold:.1f}_with_full",
        # f"per_receptive_field_unstr_trace_from_{threshold:.1f}",  # type6
        # f"per_receptive_field_unstr_trace_from_{threshold:.1f}_with_full",
        # f"type7_trace_from_{threshold:.1f}",
        # f"per_input_unstr_trace_from_{threshold:.1f}",  # type8
        # f"per_input_unstr_trace_from_{threshold:.1f}_with_full",
    ]
    for model_key, comparison_type_key, classifier in itertools.product(
        [
            "alexnet",
            # "resnet_18_cifar100",
            # "resnet_18_cifar10",
            # "densenet_cifar10",
            # "alexnet_per_channel",
            # "resnet_50",
            # "vgg_16",
        ],
        [
            "layer_num",
            # "attack_num",
            # "train_num",
            # "classifier",
            # "rank_num",
        ],
        [
            # "linear",
            "rf",
            # "ada",
            # "gb",
            # "if",
            # "ocsvm",
        ],
    ):
        plot_comparison(
            model_key=model_key,
            comparison_type_key=comparison_type_key,
            classifier=classifier,
            trace_types=trace_types,
            metric_type=metric_type,
            repeated_times=1,
            # cache=True,
        )
