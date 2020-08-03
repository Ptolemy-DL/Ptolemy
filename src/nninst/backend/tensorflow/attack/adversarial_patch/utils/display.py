import math
import time

from imagenet_stubs.imagenet_2012_labels import label_to_name

from ..constant import *
from ..model_container import _circle_mask
from .show import show


def show_patch(patch):
    circle = _circle_mask((299, 299, 3))
    show(circle * patch + (1 - circle))


def report(model, step=None, show_images=False, n=400, verbose=True, scale=(0.1, 1.0)):
    """Prints a report on how well the model is doing.
    If you want to see multiple samples, pass a positive int to show_images

    Model can be a ModelContainer instance, or a string. If it's a string, we
    lookup that model name in the MultiModel
    """
    start = time.time()
    # n examples where target was in top 5
    top_5 = 0
    # n examples where target was top 1
    wins = 0
    # n examples in total
    n_batches = int(math.ceil(float(n) / BATCH_SIZE))
    total = BATCH_SIZE * n_batches

    loss = 0

    for b in range(n_batches):
        if isinstance(model, str):
            raise RuntimeError()
            # loss_per_example, probs, patched_imgs = M.inference_batch(model, scale=scale)
        else:
            loss_per_example, probs, patched_imgs = model.inference_batch(scale=scale)

        loss += np.mean(loss_per_example)
        for i in range(BATCH_SIZE):
            top_labels = np.argsort(-probs[i])[:5]
            if TARGET_LABEL in top_labels:
                top_5 += 1
                if top_labels[0] == TARGET_LABEL:
                    wins += 1
    loss = loss / n_batches
    top_5p = int(100 * float(top_5) / total)
    winp = int(100 * float(wins) / total)

    if step is not None:
        r = "Step: {} \t".format(step)
    else:
        r = ""
    r += "LogLoss: {:.1f} \tWin Rate: {}%\t Top5: {}%\tn: {}".format(
        math.log(loss), winp, top_5p, total
    )
    if verbose:
        print(r)

    if show_images:
        if show_images is True:
            show_images = 1
        _visualize_example(patched_imgs, probs, loss_per_example, show_images)
    elapsed = time.time() - start
    return {
        "logloss": math.log(loss),
        "win": winp,
        "top5": top_5p,
        "time": elapsed,
        "loss": loss,
    }


def _visualize_example(patched_imgs, probs, loss_per_example, n_reports=1):
    print(n_reports)
    for i in range(n_reports):
        show(patched_imgs[i])

        predictions_str = ""
        top_label_ids = np.argsort(-probs[i])[:5]
        for label in top_label_ids:
            p = probs[i][label]
            name = label_to_name(label)
            if len(name) > 30:
                name = name[:27] + "..."
            if name == "toaster":
                predictions_str += "\033[1m"
            name = name.ljust(30, " ")
            predictions_str += "{} {:.2f}".format(name, p)
            if name.startswith("toaster"):
                predictions_str += "\033[0m"
            predictions_str += "\n"
        # predictions_str += "\033[1mLogLoss: {:.1f}\033[0m\n".format(math.log(loss_per_example[i]))

        print(predictions_str)


def cross_model_report(meta_model, n=100, verbose=True, scale=None):
    results = {}

    print("{:15s}\t Loss\t Win%\t Top5%\t Time".format("Model Name"))

    out_start = time.time()
    for model_name in MODEL_NAMES:
        model = meta_model.name_to_container[model_name]
        r = report(model, n=n, verbose=False, scale=scale)
        results[model_name] = r
        print(
            "{:15s}\t {:.1f}\t {:.0f}%\t {:.0f}%\t {:.0f}s".format(
                model_name, r["loss"], r["win"], r["top5"], r["time"]
            )
        )

    def _avg(name):
        xs = [r[name] for r in results.values()]
        return sum(xs) / len(xs)

    elapsed = time.time() - out_start
    print(
        "{:15s}\t {:.1f}\t {:.0f}%\t {:.0f}%\t {:.0f}s".format(
            "Average/Total", _avg("loss"), _avg("win"), _avg("top5"), elapsed
        )
    )

    return results
