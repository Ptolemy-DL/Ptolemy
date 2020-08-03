from functools import partial
from typing import List

from matplotlib.ticker import MaxNLocator
from sklearn import svm
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import Lasso, SGDClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from nninst import mode
from nninst.backend.tensorflow.attack.cdrp import (
    alexnet_imagenet_example_cdrp,
    vgg_16_imagenet_example_cdrp,
)
from nninst.backend.tensorflow.attack.common import (
    alexnet_imagenet_example_stat,
    resnet_50_imagenet_example_stat,
)
from nninst.backend.tensorflow.model import VGG16, AlexNet, ResNet50
from nninst.backend.tensorflow.model.config import ALEXNET, RESNET_50
from nninst.op import *
from nninst.plot.prelude import *
from nninst.trace import TraceKey, density_name
from nninst.utils.ray import ray_init, ray_map

if __name__ == "__main__":
    mode.local()
    # mode.debug()
    ray_init()
    threshold = 0.5
    # threshold = 1.0
    normal_example_label = -1
    adversarial_example_label = -normal_example_label
    summary_path_template = (
        lambda threshold, attack_name, metric_type, model, label: f"{model}_imagenet_{metric_type}_metrics_per_layer_{threshold:.1f}_{attack_name}{to_suffix(label)}.csv"
    )
    save_path_template = (
        lambda model, label, comparison_type, classifier: f"{model}_imagenet_roc_v2[{label}][comparison={comparison_type}][classifier={classifier}].pdf"
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
            clf_name="Linear",
        ),
        "svm": Config(clf=svm.SVC()),
        "linear_svm": Config(
            clf=svm.LinearSVC(loss="squared_hinge", penalty="l1", dual=False),
            is_linear=True,
        ),
        "lasso": Config(clf=Lasso(positive=True), is_linear=True),
        "rf": Config(
            clf=RandomForestClassifier(n_estimators=100, max_depth=None),
            is_linear=False,
            clf_name="Random forest",
        ),
        "ada": Config(clf=AdaBoostClassifier(), is_linear=False, clf_name="Adaboost"),
        "gb": Config(
            clf=GradientBoostingClassifier(),
            is_linear=False,
            clf_name="Gradient boosting",
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
            model_name="alexnet",
            model=AlexNet,
            label="import",
            cdrp_fn=alexnet_imagenet_example_cdrp,
            stat_fn=alexnet_imagenet_example_stat,
            gates={
                1: "conv2d/gate1:0",
                2: "conv2d_1/gate2:0",
                3: "conv2d_2/gate3:0",
                4: "conv2d_3/gate4:0",
                5: "conv2d_4/gate5:0",
            },
            labelrotation=60,
            figsize=(8, 5),
            x_border=[4.5],
        ),
        "alexnet_per_channel": Config(
            model_name="alexnet",
            model=AlexNet,
            label="import_per_channel",
            labelrotation=60,
            figsize=(8, 5),
            x_border=[4.5],
        ),
        "alexnet_weight": Config(
            model_name="alexnet",
            model=AlexNet,
            label="import_weight",
            labelrotation=60,
            figsize=(8, 5),
            x_border=[4.5],
        ),
        "resnet_50": Config(
            model_name="resnet_50",
            model=ResNet50,
            label=None,
            stat_fn=resnet_50_imagenet_example_stat,
            labelrotation=90,
            figsize=(8, 5),
            x_border=[0.5, 9.5, 21.5, 39.5, 48.5],
        ),
        "vgg_16": Config(
            model_name="vgg_16",
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
        "vgg": Config(attack_names=["FGSM", "FGSM_iterative_targeted"]),
        "test": Config(
            attack_names=[
                "DeepFool",
                "FGSM",
                # "FGSM_targeted",
                # "FGSM_iterative_targeted",
                "BIM",
                "JSMA",
                "CWL2",
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
        "alexnet": "basic",
        # "alexnet": "test",
        "alexnet_per_channel": "basic",
        "resnet_50": "basic",
        "vgg_16": "vgg",
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
        for attack_name in trained_attack_names:
            label = config.label
            if variant is not None:
                label = f"{label}_{variant}"
            label = f"{label}_point" if use_point else config.label
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

        clf = classifier_configs[classifier].clf
        clf.fit(train_predictions, train_labels)
        #     if classifier_configs[classifier].is_linear:
        #         clf.coef_[clf.coef_ < 0] = 0

        attack_to_auc = {}
        for attack_name in attack_names:
            labels = attack_to_test_labels[attack_name]
            predictions = attack_to_test_predictions[attack_name]
            if classifier_configs[classifier].is_linear:
                y_score = clf.decision_function(predictions)
            else:
                y_score = clf.predict_proba(predictions)[:, 1]
            fpr, tpr, thresholds = metrics.roc_curve(labels, y_score)
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
                                image_id=0,
                            )
                            for class_id in range(1000)
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
                                image_id=0,
                            )
                            for class_id in range(1000)
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

    classifiers = ["linear", "rf", "ada", "gb"]
    # classifiers = ["rf", "ada", "gb"]
    # classifiers = ["linear", "rf"]

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

    trace_type_configs = {
        "trace": Config(auc_fn=get_auc, trace_name="Ours"),
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
        "unstr_point_trace_from_0.5": Config(
            auc_fn=partial(
                get_auc, use_point=True, trace_label="unstructured_density_from_0.5"
            ),
            trace_name="UnstrP(from=0.5)",
        ),
        "unstr_trace_from_0.5": Config(
            auc_fn=partial(get_auc, trace_label="unstructured_density_from_0.5"),
            # trace_name="Unstr(from=0.5)",
            trace_name="Type B",
        ),
        "unstr_trace_from_0.5_with_full": Config(
            auc_fn=partial(
                get_auc,
                trace_label="unstructured_density_from_0.5",
                compare_with_full=True,
            ),
            trace_name="Unstr(from=0.5, vs full)",
        ),
        "per_receptive_field_unstr_trace_from_0.5_with_full": Config(
            auc_fn=partial(
                get_auc,
                trace_label="per_receptive_field_unstructured_density_from_0.5",
                compare_with_full=True,
            ),
            trace_name="UnstrRF(from=0.5, vs full)",
        ),
        "per_input_unstr_trace_from_0.5_with_full": Config(
            auc_fn=partial(
                get_auc,
                trace_label="per_input_unstructured_density_from_0.5",
                compare_with_full=True,
            ),
            trace_name="UnstrWI(from=0.5, vs full)",
        ),
        "per_receptive_field_unstr_trace_from_0.5": Config(
            auc_fn=partial(
                get_auc, trace_label="per_receptive_field_unstructured_density_from_0.5"
            ),
            # trace_name="UnstrRF(from=0.5)",
            trace_name="Type C",
        ),
        "per_input_unstr_trace_from_0.5": Config(
            auc_fn=partial(
                get_auc, trace_label="per_input_unstructured_density_from_0.5"
            ),
            # trace_name="UnstrWI(from=0.5)",
            trace_name="Type D",
        ),
    }

    def plot_comparison(
        model_key: str,
        comparison_type_key: str,
        classifier: str,
        trace_types: List[str],
        metric_type: str,
        repeated_times: int = 5,
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
        #     aucs = pd.read_csv(save_path.replace(".pdf", ".csv"))

        if len(aucs) == 0:
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
        #     print(aucs)

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
        #     ax.set_ylim(ymin=0.5)
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
        aucs = aucs.groupby([comparison_type_key, "type", "attack"]).mean()
        print(save_path)
        aucs.to_csv(ensure_dir(abspath(save_path).replace(".pdf", ".csv")))
        plt.savefig(ensure_dir(abspath(save_path)), bbox_inches="tight")

    # metric_type = "ideal"
    metric_type = "real"
    trace_types = [
        "trace",
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
        # "unstr_point_trace_from_0.5",
        # "unstr_trace_from_0.5",
        # "unstr_trace_from_0.5_with_full",
        # "per_receptive_field_unstr_trace_from_0.5",
        # "per_receptive_field_unstr_trace_from_0.5_with_full",
        # "per_input_unstr_trace_from_0.5",
        # "per_input_unstr_trace_from_0.5_with_full",
    ]
    for model_key, comparison_type_key, classifier in itertools.product(
        [
            "alexnet",
            # "alexnet_per_channel",
            # "resnet_50",
            # "vgg_16",
        ],
        [
            # "layer_num",
            # "attack_num",
            # "train_num",
            "classifier",
            # "rank_num",
        ],
        [
            # "linear",
            "rf",
            # "ada",
            # "gb",
        ],
    ):
        plot_comparison(
            model_key=model_key,
            comparison_type_key=comparison_type_key,
            classifier=classifier,
            trace_types=trace_types,
            metric_type=metric_type,
            # repeated_times=1,
        )
