from typing import Any

from operations.operation import Operation
import operations.arguments as arguments
from util.config import load_network_config
from tensorflow.keras.utils import plot_model
from network.loss import determine_loss_function
from network.metrics import get_standard_metrics
from network.model import build_model
from network.optimizer import determine_optimizer
from contextlib import redirect_stdout

class Visualize(Operation):
    """Visualize the model architecture."""

    def __call__(self, dataset: str, network: str) -> None:
        config = load_network_config(network, dataset)

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

    def get_cli_arguments(self) -> list[dict[str, Any]]:
        return [
            arguments.NETWORK_ARGUMENT,
            arguments.DATASET_ARGUMENT
        ]

