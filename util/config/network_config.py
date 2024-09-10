from dataclasses import dataclass
from typing import Union

import yaml
from yaml import SafeLoader

from util.types import OptimizerType, ModelType, BackboneType, LossType


@dataclass(slots=True)
class NetworkConfig:
    """Config dataclass, holding most hyperparameters as indicated by the user as well as some defaults."""
    id: str

    # Model info
    model: ModelType
    backbone: Union[BackboneType, None]
    use_pretrained: bool

    # Learning process info
    batch_size: int
    batch_normalization: bool
    num_epochs: int
    loss: LossType
    initial_learning_rate: float
    regularization: Union[float, None]
    dropout: Union[float, None]
    optimizer: OptimizerType

    augment_data: bool
    binarize_labels: bool

def load_network_config(config_filename: str) -> NetworkConfig:
    """Load a config YAML file. Doesn't catch input errors that might be throw due to errors."""
    with open(config_filename, 'r') as config_file:
        config_vals = yaml.load(config_file, Loader=SafeLoader)

    # Try loading the config
    config = NetworkConfig(
        id=str(config_vals['id']),
        model=ModelType(config_vals['model']),
        backbone=BackboneType(config_vals['backbone']) if config_vals.get('backbone') else None,
        use_pretrained=bool(config_vals['use_pretrained']),
        batch_size=int(config_vals['batch_size']),
        batch_normalization=bool(config_vals['batch_normalization']),
        num_epochs=int(config_vals['num_epochs']),
        loss=LossType(config_vals['loss']),
        initial_learning_rate=float(config_vals['initial_learning_rate']),
        regularization=float(config_vals['regularization']),
        dropout=float(config_vals['dropout']) if config_vals.get('dropout') is not None else None,
        optimizer=OptimizerType(config_vals['optimizer']),
        augment_data=bool(config_vals['augment_data']),
        binarize_labels=bool(config_vals['binarize_labels']),
    )

    return config
