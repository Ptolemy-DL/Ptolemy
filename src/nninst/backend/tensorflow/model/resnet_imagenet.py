# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from typing import Dict

import tensorflow as tf

from nninst.backend.tensorflow.model import resnet

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = {"train": 1281167, "validation": 50000}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 1500


###############################################################################
# Running the model
###############################################################################
class ImagenetModel(resnet.Model):
    def __init__(
        self,
        resnet_size,
        data_format=None,
        num_classes=_NUM_CLASSES,
        version=resnet.DEFAULT_VERSION,
        gated=False,
        with_gates: bool = True,
    ):
        """These are the parameters that work for Imagenet data.

        Args:
          resnet_size: The number of convolutional layers needed in the model.
          data_format: Either 'channels_first' or 'channels_last', specifying which
            data format to use when setting up the model.
          num_classes: The number of output classes needed from the model. This
            enables users to extend the same model to their own datasets.
          version: Integer representing which version of the ResNet network to use.
            See README for details. Valid values: [1, 2]
        """

        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
            final_size = 512
        else:
            bottleneck = True
            final_size = 2048

        super(ImagenetModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            second_pool_size=7,
            second_pool_stride=1,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            final_size=final_size,
            version=version,
            data_format=data_format,
            gated=gated,
            with_gates=with_gates,
        )

    def __call__(
        self,
        inputs,
        training=False,
        gate_variables: Dict[str, tf.Variable] = None,
        batch_size: int = 1,
    ):
        return super().__call__(inputs, training, gate_variables)


def _get_block_sizes(resnet_size):
    """The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = (
            "Could not find layers for selected Resnet size.\n"
            "Size received: {}; sizes allowed: {}.".format(resnet_size, choices.keys())
        )
        raise ValueError(err)
