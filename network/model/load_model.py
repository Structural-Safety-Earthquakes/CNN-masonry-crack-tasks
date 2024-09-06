import os
import sys

from tensorflow.keras.models import model_from_json
from tensorflow.keras import Model

from util.config import Config
from util.types import ModelType


def load_model(config: Config) -> Model:
    """
    Load the model indicated in the config file.
    If none are indicated, take the best performing model for the current dataset-network combination.
    """
    # Resolve missing target by taking best performing model from the weights and model folder.
    weights_file = config.weights_file
    if weights_file is None:
        candidates = [weight for weight in os.listdir(config.output_weights_dir) if weight.endswith('.h5')]
        best_value = '0000000000000'
        for candidate in candidates:
            if candidate[-6:-3] > best_value[-6:-3]:
                best_value = candidate

        if best_value != '0000000000000':
            weights_file = best_value
        else:
            raise ValueError('Please provide a valid model file or train a model using the current dataset-network combination.')

    weights_file = os.path.join(config.output_weights_dir, weights_file)
    print(f'Using model {config.output_model_file} and {weights_file} for weights.')

    # Load the model file. Some model types are exceptions, but generally we can simply load a JSON.
    if config.model == ModelType.DeepCrack:
        sys.path.append(os.path.join(os.getcwd(), 'libs', 'deepcrack'))
        from model import DeepCrack  # Requires Git submodule to be loaded

        model = DeepCrack(input_shape=(config.batch_size, *config.dataset_config.image_dims))
        model.load_weights(weights_file)
        return model

    if config.model == ModelType.DeepLabV3:
        raise NotImplementedError('DeepLabV3 is not yet implemented.')

    # Open the file and load the model and weights
    with open(config.output_model_file, 'r') as model_json_file:
        model = model_from_json(model_json_file.read())
    model.load_weights(weights_file)

    return model