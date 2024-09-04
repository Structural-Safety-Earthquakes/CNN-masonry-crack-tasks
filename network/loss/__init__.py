from argparse import ArgumentError
from typing import Callable

import tensorflow as tf

from network.loss.loss_functions import weighted_cross_entropy, f1_score_loss
from util.config import Config
from util.types import LossType


def determine_loss_function(config: Config) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Determine the loss function using the config and function specific values around it."""
    match config.loss:
        case LossType.FocalLoss:
            return tf.keras.losses.BinaryFocalCrossentropy(from_logits=False, apply_class_balancing=False, alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA)
        case LossType.BCE:
            return tf.keras.losses.BinaryCrossentropy()
        case LossType.WCE:
            return weighted_cross_entropy(config.WCE_BETA)
        case LossType.F1Score:
            return f1_score_loss(False)
        case LossType.F1ScoreDilate:
            return f1_score_loss(True)
        case _:
            raise ArgumentError(f'Unknown loss type: {config.loss}')
