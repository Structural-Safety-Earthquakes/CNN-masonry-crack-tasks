import tensorflow as tf


def dilation2d(img: tf.Tensor, kernel_size: int):
    """
    Function to dilate an image, used to dilate labels to create some tolerance.
    """
    with tf.compat.v1.variable_scope('dilation2d'):
        kernel = tf.zeros((kernel_size, kernel_size, 1))
        return tf.nn.dilation2d(
            img,
            kernel,
            (1, 1, 1, 1),
            'SAME',
            'NHWC',
            (1, 1, 1, 1)
        )