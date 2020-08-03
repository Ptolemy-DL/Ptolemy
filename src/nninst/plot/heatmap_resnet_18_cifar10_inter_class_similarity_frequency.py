import numpy as np
import pandas as pd
import seaborn as sns

from nninst.backend.tensorflow.model.resnet_18_cifar10 import ResNet18Cifar10
from nninst.backend.tensorflow.trace.resnet_18_cifar10_inter_class_similarity import (
    resnet_18_cifar10_inter_class_similarity_frequency,
)
from nninst.op import Conv2dOp, DenseOp

np.random.seed(0)
sns.set()

threshold = 0.5
frequency = int(2500 * 0.1)
label = None
variant = None
base_name = f"resnet_18_cifar10_inter_class_similarity_frequency_{frequency}"
cmap = "Greens"

same_class_similarity = []
diff_class_similarity = []
layer_names = []

layers = ResNet18Cifar10.graph().load().ops_in_layers(Conv2dOp, DenseOp)

for layer_name in [
    None,
    # *layers,
]:
    similarity = resnet_18_cifar10_inter_class_similarity_frequency(
        threshold, frequency, label, variant=variant, layer_name=layer_name
    ).load()
    same_class_similarity.append(
        np.mean(similarity[np.eye(similarity.shape[0], dtype=bool)])
    )
    diff_class_similarity.append(
        np.mean(
            similarity[
                np.tri(similarity.shape[0], similarity.shape[1], k=-1, dtype=bool)
            ]
        )
    )
    if layer_name is None:
        file_name = base_name
        layer_names.append("All")
    else:
        file_name = base_name + "_" + layer_name[: layer_name.index("/")]
        layer_names.append(layer_name[: layer_name.index("/")])
    plot_array = np.around(similarity, decimals=2)
    ax = sns.heatmap(plot_array, cmap=cmap, vmax=plot_array.max(), annot=True)
    ax.set(xlabel="Class", ylabel="Class")
    fig = ax.get_figure()
    # fig.savefig(f"{file_name}.pdf", bbox_inches="tight")
    fig.savefig(f"{file_name}.png", bbox_inches="tight")
    # np.savetxt(f"{file_name}.csv", similarity, delimiter=",")
    fig.clf()

for layer_name, similarity in zip(
    ["avg", "first_half", "second_half"],
    [
        np.mean(
            [
                resnet_18_cifar10_inter_class_similarity_frequency(
                    threshold, frequency, label, variant=variant, layer_name=layer
                ).load()
                for layer in layers
            ],
            axis=0,
        ),
        np.mean(
            [
                resnet_18_cifar10_inter_class_similarity_frequency(
                    threshold, frequency, label, variant=variant, layer_name=layer
                ).load()
                for layer in layers[: len(layers) // 2]
            ],
            axis=0,
        ),
        np.mean(
            [
                resnet_18_cifar10_inter_class_similarity_frequency(
                    threshold, frequency, label, variant=variant, layer_name=layer
                ).load()
                for layer in layers[len(layers) // 2 :]
            ],
            axis=0,
        ),
    ],
):
    file_name = base_name + "_" + layer_name
    plot_array = np.around(similarity, decimals=2)
    ax = sns.heatmap(plot_array, cmap=cmap, vmax=plot_array.max(), annot=True)
    ax.set(xlabel="Class", ylabel="Class")
    fig = ax.get_figure()
    # fig.savefig(f"{file_name}.pdf", bbox_inches="tight")
    fig.savefig(f"{file_name}.png", bbox_inches="tight")
    # np.savetxt(f"{file_name}.csv", similarity, delimiter=",")
    fig.clf()

summary_df = pd.DataFrame(
    {
        "Same Class": same_class_similarity,
        "Diff Class": diff_class_similarity,
        "Layer": layer_names,
    }
)
summary_df.to_csv(f"{base_name}_summary.csv", index=False)
