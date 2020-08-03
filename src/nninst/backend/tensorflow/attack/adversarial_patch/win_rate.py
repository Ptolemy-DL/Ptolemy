import math
import time

import matplotlib.pyplot as plt
import numpy as np

from .utils.display import report

areas_to_report = list(np.linspace(0.01, 0.10, 10)) + [
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]


def calculate_win_rates(models, eval_samples_per_scale=100):
    start = time.time()
    rows = len(models)
    results = np.zeros((rows, len(areas_to_report)))
    for (i, model) in enumerate(models):
        print("Evaluating %s" % model.model_name)
        for (j, a) in enumerate(areas_to_report):
            sc = 2 * math.sqrt(a / math.pi)
            win = report(model, scale=sc, verbose=False, n=eval_samples_per_scale)[
                "win"
            ]
            results[i, j] = win
    print("Calculated wins in {:.0f}s".format(time.time() - start))
    return results


def plot_win_rates(wins, labels, title):
    assert wins.shape[0] == len(labels)
    for (i, l) in enumerate(labels):
        plt.plot([a * 100.0 for a in areas_to_report], wins[i], label=l)
    plt.title(title)
    plt.legend()
    plt.xlabel("Attack as % of image size")
    plt.ylabel("Attack success rate")

    plt.show()
