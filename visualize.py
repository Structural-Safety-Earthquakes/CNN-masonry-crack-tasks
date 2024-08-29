import os

from tensorflow.keras.utils import plot_model

from loss_class import Loss
from metrics_class import Metrics
from network_class import Network
from optimizer_class import Optimizer
from util.config import Config
from contextlib import redirect_stdout

from util.types import LossType


def visualize_architecture(config: Config):
    """Create plots of the network architecture."""
    # TODO: remove args by refactoring dependencies
    match config.loss:
        case LossType.FocalLoss:
            loss_str = 'Focal_Loss'
        case LossType.WCE:
            loss_str = 'WCE'
        case LossType.BCE:
            loss_str = 'Binary_Crossentropy'
        case LossType.F1Score:
            loss_str = 'F1_score_Loss'
        case LossType.F1ScoreDilate:
            loss_str = 'F1_score_Loss_dill'
        case _:
            loss_str = 'WCE'
    args = {
        'main': os.getcwd(),
        'loss': loss_str,
        'focal_loss_a': config.FOCAL_LOSS_ALPHA,
        'focal_loss_g': config.FOCAL_LOSS_GAMMA,
        'WCE_beta': config.WCE_BETA,
        'opt': config.optimizer.value,
        'regularization': config.regularization,
        'model': f'sm_{config.model.value}_{config.backbone.value}' if config.backbone is not None else config.model.value,
        'dropout': config.dropout,
        'batchnorm': config.batch_normalization,
        'init': config.UNET_LAYER_WEIGHT_INITIALIZERS.value,
        'encoder_weights': 'imagenet' if config.use_pretrained else None
    }

    # %%
    # Prepare model for training
    #
    model = Network(
        args,
        config.image_dims,
        config.UNET_NUM_FILTERS,
        config.batch_size,
        config.initial_learning_rate,
        Optimizer(args, config.initial_learning_rate).define_Optimizer(),
        Loss(args).define_Loss(),
        Metrics(args).define_Metrics()
    ).define_Network()

    # Create a visual plot
    plot_model(model, to_file=config.output_network_figure_file, show_shapes=True)

    # Create a txt summary
    with open(config.output_txt_summary_file, 'w') as f:
        with redirect_stdout(f):
            model.summary()
