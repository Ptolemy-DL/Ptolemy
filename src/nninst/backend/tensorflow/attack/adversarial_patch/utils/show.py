import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if "MPLBACKEND" not in os.environ:
    print("use Agg backend")
    matplotlib.use("Agg")


def _convert(im):
    return ((im + 1) * 127.5).astype(np.uint8)


def show(im):
    plt.axis("off")
    plt.imshow(_convert(im), interpolation="nearest")
    plt.show()
