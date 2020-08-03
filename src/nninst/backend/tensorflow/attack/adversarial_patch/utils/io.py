import os.path as osp
import pickle
from datetime import datetime

import numpy as np
import PIL.Image

from ..constant import DATA_DIR
from .display import show_patch
from .show import _convert


def save_obj(obj, file_name):
    serialized = pickle.dumps(obj, protocol=0)
    dest_file = osp.join(DATA_DIR, file_name)
    with open(dest_file, "wb") as f:
        f.write(serialized)


def load_obj(file_name):
    dest_file = osp.join(DATA_DIR, file_name)
    with open(dest_file, "rb") as f:
        pkl = f.read()
    return pickle.loads(pkl)


def _latest_snapshot_path(experiment_name):
    """Return the latest pkl file"""
    return osp.join(DATA_DIR, "%s.latest" % (experiment_name))


def _timestamped_snapshot_path(experiment_name):
    """Return a timestamped pkl file"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return osp.join(DATA_DIR, "%s.%s" % (experiment_name, timestamp))


def save_patch(experiment_name, model):
    """Save a snapshot for the given experiment"""

    def _serialize_patch(dest_file):
        patch = model.patch()
        serialized = pickle.dumps(patch, protocol=0)  # protocol 0 is printable ASCII

        with open(dest_file + ".pkl", "wb") as f:
            f.write(serialized)
            print("Wrote patch to %s" % dest_file)
        with open(dest_file + ".jpg", "wb") as f:
            PIL.Image.fromarray(_convert(model.patch())).save(f, "JPEG")

    _serialize_patch(_latest_snapshot_path(experiment_name))
    _serialize_patch(_timestamped_snapshot_path(experiment_name))


def load_patch(experiment_name_or_patch_file, model, dontshow=False):
    if experiment_name_or_patch_file.startswith(DATA_DIR):
        patch_file = experiment_name_or_patch_file
    else:
        patch_file = _latest_snapshot_path(experiment_name_or_patch_file)
    with open(patch_file + ".pkl", "rb") as f:
        pkl = f.read()
    patch = pickle.loads(pkl)
    model.patch(patch)
    if not dontshow:
        show_patch(patch)


def get_im(path):
    with open(osp.join(DATA_DIR, path), "rb") as f:
        pic = PIL.Image.open(f)
        pic = pic.resize((299, 299), PIL.Image.ANTIALIAS)
        if path.endswith(".png"):
            ch = 4
        else:
            ch = 3
        pic = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], ch)[:, :, :3]
        pic = pic / 127.5 - 1
    return pic
