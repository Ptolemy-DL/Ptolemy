from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

import numpy as np
import PIL
from PIL.Image import Image

__all__ = ["Dataset"]

ImageId = TypeVar("ImageId")


class Dataset(ABC, Generic[ImageId]):
    @property
    @abstractmethod
    def size(self) -> int:
        ...

    def image(self, image_id: ImageId) -> Image:
        return PIL.Image.fromarray(self.image_np(image_id))

    def image_np(self, image_id: ImageId) -> np.ndarray:
        return np.array(self.image(image_id))

    @abstractmethod
    def label(self, image_id: ImageId) -> int:
        ...

    def image_with_label(self, image_id: ImageId) -> Tuple[Image, int]:
        return self.image(image_id), self.label(image_id)

    def __getitem__(self, image_id: ImageId) -> Tuple[Image, int]:
        return self.image_with_label(image_id)

    def __len__(self) -> int:
        return self.size
