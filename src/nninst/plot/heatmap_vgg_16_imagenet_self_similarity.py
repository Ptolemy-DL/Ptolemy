import numpy as np
import seaborn as sns

np.random.seed(0)

sns.set()
my_data = np.genfromtxt("vgg_16_imagenet_self_similarity.csv", delimiter=",")
ax = sns.heatmap(my_data, cmap="Greys", vmax=0.65, annot=True)
ax.set(xlabel="Class", ylabel="Class")
fig = ax.get_figure()
fig.savefig("vgg_16_imagenet_self_similarity.pdf", bbox_inches="tight")
fig.savefig("vgg_16_imagenet_self_similarity.png", bbox_inches="tight")
