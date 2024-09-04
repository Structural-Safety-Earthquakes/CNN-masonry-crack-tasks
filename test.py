import os
from pathlib import Path

from evaluate_class import LoadModel
from subroutines.HDF5 import HDF5DatasetGeneratorMask
from subroutines.visualize_predictions import Visualize_Predictions
from util.config import Config


def generate_predictions(config: Config):
    """Generate the predictions given a specific configuration."""

    if config.prediction_file is None:
        candidates = [weight for weight in os.listdir(config.output_weights_dir) if weight.endswith('.h5')] + [model for model in os.listdir(config.output_models_dir) if model.endswith('.h5')]
        best_value = '0000000000000'
        for candidate in candidates:
            if candidate[-6:-3] > best_value[-6:-3]:
                best_value = candidate

        config.prediction_file = best_value if best_value != '0000000000000' else None
        print(f'Using {config.prediction_file} for predictions.')

    # TODO: remove args by refactoring dependencies
    args = {
        'main': os.getcwd(),
        'model': config.model.value,
        'weights': config.output_weights_dir + os.sep,
        'pretrained_filename': config.prediction_file,
        'save_model_weights': 'model' if config.save_model else 'weights',
        'model_json': config.output_model_file,
        'EVAL_HDF5': config.dataset_config.dataset_validation_set_file,
        'predictions_subfolder': config.output_predictions_dir + os.sep,
        'predictions_dilate': config.dilate_labels
    }
    model = LoadModel(args, config.dataset_config.image_dims, config.batch_size).load_pretrained_model()

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
