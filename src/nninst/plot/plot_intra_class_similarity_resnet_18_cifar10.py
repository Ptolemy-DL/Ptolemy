import numpy as np
import pandas as pd
import seaborn as sns

from nninst.backend.tensorflow.attack.common import intra_class_similarity

threshold = 0.5
attack_name = "original"

name = "resnet_18_cifar10"
label = None
class_to_similarity = {}
for class_id in range(10):
    similarity = intra_class_similarity(
        name=name,
        threshold=threshold,
        class_id=class_id,
        attack_name=attack_name,
        label=label,
        example_trace_fn=None,
        image_ids=None,
    ).load()
    class_to_similarity[str(class_id)] = similarity[
        np.tri(similarity.shape[0], similarity.shape[1], k=-1, dtype=bool)
    ]
df = pd.DataFrame(class_to_similarity)
df.info()

similarity_col = np.concatenate([df[str(class_id)] for class_id in range(10)])
class_col = np.concatenate(
    [[str(class_id)] * len(df[str(class_id)]) for class_id in range(10)]
)
plot_df = pd.DataFrame({"Similarity": similarity_col, "Class": class_col})
plot_df.info()

ax = sns.boxplot(x="Class", y="Similarity", data=plot_df)
fig = ax.get_figure()
fig.savefig("intra_class_similarity_resnet_18_cifar10.pdf", bbox_inches="tight")
fig.savefig("intra_class_similarity_resnet_18_cifar10.png", bbox_inches="tight")

summary_df = plot_df.groupby("Class").mean().reset_index()
summary_df.to_csv("intra_class_similarity_resnet_18_cifar10.csv", index=False)
