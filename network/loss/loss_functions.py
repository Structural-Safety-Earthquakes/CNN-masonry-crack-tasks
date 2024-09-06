from typing import Callable

import tensorflow as tf
import tensorflow.keras.backend as K

from util.image_operations import dilation2d


def weighted_cross_entropy(beta: float) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Weighted Cross-Entropy (WCE) Loss.
    It is based on the implementation found in the link below:
    https://jeune-research.tistory.com/entry/Loss-Functions-for-Image-Segmentation-Distribution-Based-Losses
    """
    def loss_function(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Convert to logits
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        y_pred = tf.math.log(y_pred / (1 - y_pred))

        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=beta)
        return tf.reduce_mean(loss)
    return loss_function

def f1_score_loss(dilate: bool = False, smooth: float = 1.) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    The F1-score, same as the one used for the metrics but adjusted to return a Tensor.
    We subtract it by one since we aim for F1-score maximization and loss minimization.
    """
    def loss_function(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        flat_y_true = K.flatten(y_true)
        flat_y_pred = K.flatten(y_pred)

        intersection = K.flatten(dilation2d(y_true, 5)) * flat_y_pred if dilate else flat_y_true * flat_y_pred
        score = (2. * K.sum(intersection) + smooth) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smooth)
        return 1. - score
    return loss_function
