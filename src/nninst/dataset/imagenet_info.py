import glob
import os
from typing import Tuple

from PIL import Image

from nninst.dataset import Dataset
from nninst.dataset.envs import IMAGENET_RAW_DIR

__all__ = ["train", "test"]


class ImageNetDataset(Dataset[Tuple[int, int]]):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    @property
    def size(self) -> int:
        raise NotImplementedError()

    def image_path(self, image_id: Tuple[int, int]) -> str:
        class_id = image_id[0]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        labels_file = "imagenet_lsvrc_2015_synsets.txt"
        challenge_synsets = [
            l.strip()
            for l in open(os.path.join(current_dir, labels_file), "r").readlines()
        ]
        jpeg_file_path = "%s/%s/*.JPEG" % (
            self.dataset_dir,
            challenge_synsets[class_id],
        )
        matching_files = glob.glob(jpeg_file_path)
        return matching_files[image_id[1]]

    def image(self, image_id: Tuple[int, int]) -> Image.Image:
        return Image.open(self.image_path(image_id))

    def label(self, image_id: Tuple[int, int]) -> int:
        return image_id[0]

    def image_with_label(self, image_id: Tuple[int, int]) -> Tuple[Image.Image, int]:
        return self.image(image_id), self.label(image_id)


def train() -> ImageNetDataset:
    return ImageNetDataset(f"{IMAGENET_RAW_DIR}/train/")


def test() -> ImageNetDataset:
    return ImageNetDataset(f"{IMAGENET_RAW_DIR}/validation/")
