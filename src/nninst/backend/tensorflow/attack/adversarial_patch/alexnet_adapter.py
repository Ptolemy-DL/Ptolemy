import tensorflow as tf

from nninst.backend.tensorflow.model.config import ALEXNET
from nninst.backend.tensorflow.utils import restore_scope
from nninst.utils.fs import abspath


def alexnet_adapter(sess, input_tensor, weights="imagenet"):
    assert weights in {"imagenet", None}
    with restore_scope(
        sess, tf.train.latest_checkpoint(abspath(ALEXNET.model_dir + "_import"))
    ):
        logits = ALEXNET.network_class()(input_tensor)
        return logits
