import os
from dataclasses import dataclass

import yaml
from yaml import SafeLoader

# Path constants, change at your own leisure
DATASETS_ROOT_DIR: str = 'dataset'
DATASETS_IMAGE_DIR: str = 'images'
DATASETS_LABEL_DIR: str = 'labels'
TRAIN_DATASET_FILE: str = 'train.hdf5'
VALIDATION_DATASET_FILE: str = 'validation.hdf5'

@dataclass(slots=True)
class DatasetConfig:
    """Config dataclass, holding all dataset specific parameters."""
    dataset_dir: str
    image_dims: tuple[int, int, int]
    validation_split_percent: float

    dataset_images_dir: str # Inferred, uses dataset_dir
    dataset_labels_dir: str # Inferred, uses dataset_dir
    dataset_train_set_file: str # Inferred, uses dataset_dir
    dataset_validation_set_file: str # Inferred, uses dataset_dir

def load_data_config(filename: str) -> DatasetConfig:
    """Load a config YAML file. Doesn't catch input errors that might be throw due to errors."""
    with open(filename, 'r') as config_file:
        config_vals = yaml.load(config_file, Loader=SafeLoader)

    # Try loading the config
    return DatasetConfig(
        dataset_dir=str(config_vals['dataset_dir']),
        image_dims=tuple(config_vals['image_dims']),
        validation_split_percent=float(config_vals['validation_split_percent']),
        dataset_images_dir=os.path.join(DATASETS_ROOT_DIR, config_vals['dataset_dir'], DATASETS_IMAGE_DIR),
        dataset_labels_dir=os.path.join(DATASETS_ROOT_DIR, config_vals['dataset_dir'], DATASETS_LABEL_DIR),
        dataset_train_set_file=os.path.join(DATASETS_ROOT_DIR, config_vals['dataset_dir'], TRAIN_DATASET_FILE),
        dataset_validation_set_file=os.path.join(DATASETS_ROOT_DIR, config_vals['dataset_dir'], VALIDATION_DATASET_FILE),
    )
