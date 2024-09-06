import tensorflow as tf
import tensorflow.keras.backend as K

from util.image_operations import dilation2d

DILATION_KERNEL_SIZE = 5

def recall(y_true: tf.Tensor, y_pred: tf.Tensor, dilate=False) -> float:
    """
    Recall = TP / (TP + FN). Dilate label if requested.
    """
    if dilate:
        y_true = dilation2d(y_true, DILATION_KERNEL_SIZE)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    gt_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (gt_positives + K.epsilon())

def precision(y_true: tf.Tensor, y_pred: tf.Tensor, dilate=False) -> float:
    """
    Precision = TP / (TP + FP). Dilate label if requested.
    """
    if dilate:
        y_true = dilation2d(y_true, DILATION_KERNEL_SIZE)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def f1_score(y_true: tf.Tensor, y_pred: tf.Tensor, dilate=False) -> float:
    precision_val = precision(y_true, y_pred, dilate)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))

# Macros for Keras metrics. Don't use the functions aside from that
def precision_dilated(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Keras macro"""
    return precision(y_true, y_pred, True)

def f1_score_dilated(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Keras macro"""
    return f1_score(y_true, y_pred, True)
