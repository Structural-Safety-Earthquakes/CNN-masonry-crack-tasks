import os

from tensorflow.keras.utils import plot_model

from network.loss import determine_loss_function
from network.metrics import get_standard_metrics
from network_class import Network
from optimizer_class import Optimizer
from util.config import Config
from contextlib import redirect_stdout


def visualize_architecture(config: Config):
    """Create plots of the network architecture."""
    # TODO: remove args by refactoring dependencies
    args = {
        'main': os.getcwd(),
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
        config.dataset_config.image_dims,
        config.UNET_NUM_FILTERS,
        config.batch_size,
        config.initial_learning_rate,
        Optimizer(args, config.initial_learning_rate).define_Optimizer(),
        determine_loss_function(config),
        get_standard_metrics()
    ).define_Network()

    # Create a visual plot
    plot_model(model, to_file=config.output_network_figure_file, show_shapes=True)

    # Create a txt summary
    with open(config.output_txt_summary_file, 'w') as f:
        with redirect_stdout(f):
            model.summary()
