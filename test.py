import os
from pathlib import Path

from network.model import load_model
from subroutines.HDF5 import HDF5DatasetGeneratorMask
from subroutines.visualize_predictions import Visualize_Predictions
from util.config import Config


def generate_predictions(config: Config):
    """Generate the predictions given a specific configuration."""
    # TODO: remove args by refactoring dependencies
    args = {
        'main': os.getcwd(),
        'EVAL_HDF5': config.dataset_config.dataset_validation_set_file,
        'predictions_subfolder': config.output_predictions_dir + os.sep,
        'predictions_dilate': config.dilate_labels
    }
    model = load_model(config)

    # Do not use data augmentation when evaluating model: aug=None
    eval_gen = HDF5DatasetGeneratorMask(
        config.dataset_config.dataset_validation_set_file,
        config.batch_size,
        aug=None,
        shuffle=False,
        binarize=config.binarize_labels
    )

    # Use the pretrained model to generate predictions for the input samples from a data generator
    predictions = model.predict(
        eval_gen.generator(),
        steps=eval_gen.numImages // config.batch_size + 1,
        max_queue_size=config.batch_size * 2,
        verbose=1
    )

    # Visualize  predictions
    # Create a plot with original image, ground truth and prediction
    # Show the metrics for the prediction
    # Output will be stored in the predictions folder
    Path(config.output_predictions_dir).parent.mkdir(exist_ok=True)
    Visualize_Predictions(args, predictions)
