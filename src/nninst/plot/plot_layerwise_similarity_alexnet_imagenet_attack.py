import numpy as np
import pandas as pd
import seaborn as sns

from nninst.backend.tensorflow.model import AlexNet
from nninst.op import Conv2dOp, DenseOp
from nninst.utils.fs import abspath

threshold = 0.5

path_template = "alexnet_imagenet_real_metrics_per_layer_{0:.1f}_{1}_{2}.csv"
attack_name = "FGSM"
label_name = "import_rank1"
path = "metrics/" + path_template.format(threshold, attack_name, label_name,)
df = pd.read_csv(abspath(path))
df.info()

layers = AlexNet.graph().load().ops_in_layers(Conv2dOp, DenseOp)
for layer in layers:
    df[f"{layer}.similarity"] = (
        df[f"{layer}.overlap_size_in_class"] / df[f"{layer}.overlap_size_total"]
    )
df.info()

similarity_col = np.concatenate([df[f"{layer}.similarity"] for layer in layers])
layer_col = np.concatenate(
    [[layer[: layer.index("/")]] * len(df[f"{layer}.similarity"]) for layer in layers]
)
plot_df = pd.DataFrame({"Similarity": similarity_col, "Layer": layer_col})
plot_df.info()

ax = sns.boxplot(x="Layer", y="Similarity", data=plot_df)
ax.tick_params(axis="x", labelrotation=45)
fig = ax.get_figure()
fig.savefig(
    f"layerwise_similarity_alexnet_imagenet_{attack_name}.pdf", bbox_inches="tight"
)
fig.savefig(
    f"layerwise_similarity_alexnet_imagenet_{attack_name}.png", bbox_inches="tight"
)

summary_df = plot_df.groupby("Layer").mean().reset_index()
summary_df.to_csv(
    f"layerwise_similarity_alexnet_imagenet_{attack_name}.csv", index=False
)
