import os
from pathlib import Path
from typing import Any

from operations.operation import Operation
import operations.arguments as arguments
from network.model import load_model
from subroutines.HDF5 import HDF5DatasetGeneratorMask
from subroutines.visualize_predictions import Visualize_Predictions
from util.config import load_network_config, load_data_config


class Test(Operation):
    """Operation which tests a trained network"""

    def __call__(self, dataset: str, network: str) -> None:
        """Generate the predictions given a specific configuration."""
        network_config = load_network_config(network, dataset)
        dataset_config = load_data_config(dataset)

        # TODO: remove args by refactoring dependencies
        args = {
            'main': os.getcwd(),
            'EVAL_HDF5': dataset_config.dataset_validation_set_file,
            'predictions_subfolder': network_config.output_predictions_dir + os.sep,
            'predictions_dilate': network_config.dilate_labels
        }
        model = load_model(network_config)

        # Do not use data augmentation when evaluating model: aug=None
        eval_gen = HDF5DatasetGeneratorMask(
            dataset_config.dataset_validation_set_file,
            network_config.batch_size,
            aug=None,
            shuffle=False,
            binarize=network_config.binarize_labels
        )

        # Use the pretrained model to generate predictions for the input samples from a data generator
        predictions = model.predict(
            eval_gen.generator(),
            steps=eval_gen.numImages // network_config.batch_size + 1,
            max_queue_size=network_config.batch_size * 2,
            verbose=1
        )

        # Visualize  predictions
        # Create a plot with original image, ground truth and prediction
        # Show the metrics for the prediction
        # Output will be stored in the predictions folder
        Path(network_config.output_predictions_dir).parent.mkdir(exist_ok=True)
        Visualize_Predictions(args, predictions)

    def get_cli_arguments(self) -> list[dict[str, Any]]:
        return [
            arguments.NETWORK_ARGUMENT,
            arguments.DATASET_ARGUMENT,
        ]
