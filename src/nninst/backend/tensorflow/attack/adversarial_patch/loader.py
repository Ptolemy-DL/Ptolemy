import random

import numpy as np
import PIL.Image

from nninst.dataset import imagenet_info

from .constant import BATCH_SIZE


def load_image(image_path):
    im = PIL.Image.open(image_path)
    im = im.resize((299, 299), PIL.Image.ANTIALIAS)
    if image_path.endswith(".png"):
        ch = 4
    else:
        ch = 3
    image_data = np.array(im.getdata())
    if image_data.size == np.prod(im.size):
        im = np.repeat(image_data.reshape(im.size[0], im.size[1], 1), 3, axis=2)
    else:
        im = image_data.reshape(im.size[0], im.size[1], ch)[:, :, :3]
    return im / 127.5 - 1
    # return im.astype(np.float)


class StubImageLoader:
    """An image loader that uses just a few ImageNet-like images.
    In the actual paper, we used real ImageNet images, but we can't include them
    here because of licensing issues.
    """

    def __init__(self):
        self.toaster_image = None
        imagenet_test = imagenet_info.test()
        self.image_paths = [
            imagenet_test.image_path((class_id, 0)) for class_id in range(1000)
        ]
        # self.image_paths = imagenet_stubs.get_image_paths()
        toaster_image_path = list(
            filter(
                lambda image_path: image_path.endswith("toaster.jpg"), self.image_paths
            )
        )
        self.image_paths = list(
            filter(
                lambda image_path: not image_path.endswith("toaster.jpg"),
                self.image_paths,
            )
        )
        self.images = [None] * len(self.image_paths)
        if len(toaster_image_path) > 0:
            self.toaster_image = load_image(toaster_image_path[0])

    def get_image(self, index):
        if self.images[index] is None:
            self.images[index] = load_image(self.image_paths[index])
        return self.images[index]

    def get_images(self):
        return list(
            map(
                self.get_image,
                random.sample(list(range(len(self.image_paths))), BATCH_SIZE),
            )
        )
        # return list(map(self.get_image, list(range(len(self.image_paths)))[:BATCH_SIZE]))


image_loader = StubImageLoader()
