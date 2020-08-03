import numpy as np
import pandas as pd
import seaborn as sns

from nninst.backend.tensorflow.model import AlexNet
from nninst.backend.tensorflow.trace.alexnet_imagenet_inter_class_similarity import (
    alexnet_imagenet_inter_class_similarity,
)
from nninst.op import Conv2dOp, DenseOp

np.random.seed(0)
sns.set()

threshold = 0.5
label = "import"
variant = None
base_name = "alexnet_imagenet_inter_class_similarity"
cmap = "Greens"

same_class_similarity = []
diff_class_similarity = []
layer_names = []

layers = AlexNet.graph().load().ops_in_layers(Conv2dOp, DenseOp)

for layer_name in [
    None,
    *layers,
]:
    similarity = alexnet_imagenet_inter_class_similarity(
        threshold, label, variant=variant, layer_name=layer_name
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
    ["first_half", "second_half"],
    [
        np.mean(
            [
                alexnet_imagenet_inter_class_similarity(
                    threshold, label, variant=variant, layer_name=layer
                ).load()
                for layer in layers[: len(layers) // 2]
            ],
            axis=0,
        ),
        np.mean(
            [
                alexnet_imagenet_inter_class_similarity(
                    threshold, label, variant=variant, layer_name=layer
                ).load()
                for layer in layers[len(layers) // 2 :]
            ],
            axis=0,
        ),
        *layers,
    ],
):
    file_name = base_name + "_" + layer_name
    plot_array = np.around(similarity, decimals=2)
    ax = sns.heatmap(plot_array, cmap=cmap, vmax=plot_array.max(), annot=True)
    ax.set(xlabel="Class", ylabel="Class")
    fig = ax.get_figure()
    fig.savefig(f"{file_name}.pdf", bbox_inches="tight")
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
