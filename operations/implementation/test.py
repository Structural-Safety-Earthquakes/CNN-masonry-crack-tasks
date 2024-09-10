import os
from typing import Any, Union

from operations.operation import Operation
import operations.arguments as arguments
from network.model import load_model
from subroutines.HDF5 import HDF5DatasetGeneratorMask
from subroutines.visualize_predictions import Visualize_Predictions
from util.config import load_network_config, load_data_config, load_output_config


class Test(Operation):
    """Operation which tests a trained network"""

    def __call__(self, dataset: str, network: str, weights: Union[str, None], dilate: bool) -> None:
        """Generate the predictions given a specific configuration."""
        network_config = load_network_config(network)
        dataset_config = load_data_config(dataset)
        output_config = load_output_config(network_id=network_config.id, dataset_id=dataset_config.dataset_dir)

        # TODO: remove args by refactoring dependencies
        args = {
            'main': os.getcwd(),
            'EVAL_HDF5': output_config.validation_set_file,
            'predictions_subfolder': output_config.predictions_dir + os.sep,
            'predictions_dilate': dilate
        }
        model = load_model(network_config, output_config, dataset_config.image_dims, weights)

        # Do not use data augmentation when evaluating model: aug=None
        eval_gen = HDF5DatasetGeneratorMask(
            output_config.validation_set_file,
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
        Visualize_Predictions(args, predictions)

    def get_cli_arguments(self) -> list[dict[str, Any]]:
        return [
            arguments.NETWORK_ARGUMENT,
            arguments.DATASET_ARGUMENT,
            arguments.WEIGHTS_FILE_ARGUMENT,
            arguments.DILATE_VALIDATION_LABELS_ARGUMENT,
        ]
