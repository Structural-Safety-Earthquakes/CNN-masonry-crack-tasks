from dataclasses import dataclass

import yaml
from yaml import SafeLoader


@dataclass(slots=True)
class DatasetConfig:
    """Config dataclass, holding all dataset specific parameters."""
    dataset_dir: str
    image_dims: tuple[int, int, int]
    validation_split_percent: float

def load_data_config(filename: str) -> DatasetConfig:
    """Load a config YAML file. Doesn't catch input errors that might be throw due to errors."""
    with open(filename, 'r') as config_file:
        config_vals = yaml.load(config_file, Loader=SafeLoader)

    # Try loading the config
    return DatasetConfig(
        dataset_dir=str(config_vals['dataset_dir']),
        image_dims=tuple(config_vals['image_dims']),
        validation_split_percent=float(config_vals['validation_split_percent'])
    )
