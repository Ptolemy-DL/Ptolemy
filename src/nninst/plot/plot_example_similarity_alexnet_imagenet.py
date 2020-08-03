import numpy as np
import pandas as pd
import seaborn as sns

from nninst.backend.tensorflow.attack.common import example_similarity

threshold = 0.5
attack_name = "original"

name = "alexnet_imagenet"
label = "import"
class_to_similarity = {}
for class_id in range(0, 1000, 100):
    similarity = example_similarity(
        name=name,
        threshold=threshold,
        class_id=class_id,
        attack_name=attack_name,
        label=label,
        example_trace_fn=None,
        class_trace_fn=None,
        image_ids=None,
    ).load()
    class_to_similarity[class_id] = similarity

similarity_col = np.concatenate(
    [class_to_similarity[class_id] for class_id in range(0, 1000, 100)]
)
class_col = np.concatenate(
    [
        [str(class_id)] * len(class_to_similarity[class_id])
        for class_id in range(0, 1000, 100)
    ]
)
plot_df = pd.DataFrame({"Similarity": similarity_col, "Class": class_col})
plot_df.info()

ax = sns.boxplot(x="Class", y="Similarity", data=plot_df)
ax.set_ylim(bottom=0, top=1)
fig = ax.get_figure()
# fig.savefig("example_similarity_alexnet_imagenet.pdf", bbox_inches="tight")
fig.savefig("example_similarity_alexnet_imagenet.png", bbox_inches="tight")

summary_df = plot_df.groupby("Class").agg(["mean", "std"]).reset_index()
summary_df.to_csv("example_similarity_alexnet_imagenet.csv", index=False)
