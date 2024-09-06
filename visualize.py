import os

from tensorflow.keras.utils import plot_model

from network.loss import determine_loss_function
from network.metrics import get_standard_metrics
from network.model import build_model
from network.optimizer import determine_optimizer
from util.config import Config
from contextlib import redirect_stdout


def visualize_architecture(config: Config):
    """Create plots of the network architecture."""

    # %%
    # Prepare model for training
    #
    model = build_model(config)
    model.compile(
        optimizer=determine_optimizer(config),
        loss=determine_loss_function(config),
        metrics=[get_standard_metrics()]
    )

    # Create a visual plot
    plot_model(model, to_file=config.output_network_figure_file, show_shapes=True)

    # Create a txt summary
    with open(config.output_txt_summary_file, 'w') as f:
        with redirect_stdout(f):
            model.summary()
