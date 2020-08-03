import math

import numpy as np
import tensorflow as tf

from nninst.backend.tensorflow.attack.adversarial_patch.utils.show import show


def _transform_vector(width, x_shift, y_shift, im_scale, rot_in_degrees):
    """
     If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1],
     then it maps the output point (x, y) to a transformed input point
     (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
     where k = c0 x + c1 y + 1.
     The transforms are inverted compared to the transform mapping input points to output points.
    """

    rot = float(rot_in_degrees) / 90.0 * (math.pi / 2)

    # Standard rotation matrix
    # (use negative rot because tf.contrib.image.transform will do the inverse)
    rot_matrix = np.array(
        [[math.cos(-rot), -math.sin(-rot)], [math.sin(-rot), math.cos(-rot)]]
    )

    # Scale it
    # (use inverse scale because tf.contrib.image.transform will do the inverse)
    inv_scale = 1.0 / im_scale
    xform_matrix = rot_matrix * inv_scale
    a0, a1 = xform_matrix[0]
    b0, b1 = xform_matrix[1]

    # At this point, the image will have been rotated around the top left corner,
    # rather than around the center of the image.
    #
    # To fix this, we will see where the center of the image got sent by our transform,
    # and then undo that as part of the translation we apply.
    x_origin = float(width) / 2
    y_origin = float(width) / 2

    x_origin_shifted, y_origin_shifted = np.matmul(
        xform_matrix, np.array([x_origin, y_origin])
    )

    x_origin_delta = x_origin - x_origin_shifted
    y_origin_delta = y_origin - y_origin_shifted

    # Combine our desired shifts with the rotation-induced undesirable shift
    a2 = x_origin_delta - (x_shift / (2 * im_scale))
    b2 = y_origin_delta - (y_shift / (2 * im_scale))

    # Return these values in the order that tf.contrib.image.transform expects
    return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)


def test_random_transform(min_scale=0.5, max_scale=1.0, max_rotation=22.5):
    """
    Scales the image between min_scale and max_scale
    """
    img_shape = [100, 100, 3]
    img = np.ones(img_shape)

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    image_in = tf.placeholder(dtype=tf.float32, shape=img_shape)
    width = img_shape[0]

    def _random_transformation():
        im_scale = np.random.uniform(low=min_scale, high=1.0)

        padding_after_scaling = (1 - im_scale) * width
        x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
        y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)

        rot = np.random.uniform(-max_rotation, max_rotation)

        return _transform_vector(
            width,
            x_shift=x_delta,
            y_shift=y_delta,
            im_scale=im_scale,
            rot_in_degrees=rot,
        )

    random_xform_vector = tf.py_func(_random_transformation, [], tf.float32)
    random_xform_vector.set_shape([8])

    output = tf.contrib.image.transform(image_in, random_xform_vector, "BILINEAR")

    xformed_img = sess.run(output, feed_dict={image_in: img})

    show(xformed_img)


if __name__ == "__main__":
    for i in range(2):
        print("Test image with random transform: %s" % (i + 1))
        test_random_transform(min_scale=0.25, max_scale=2.0, max_rotation=22.5)
