import os
from dataclasses import dataclass
from typing import Union

import yaml
from yaml import SafeLoader

from util.types import OptimizerType, ModelType, BackboneType, UnetWeightInitializerType, LossType

# Path constants, change at your own leisure
DATASETS_ROOT_DIR: str = 'datasets'
DATASETS_PROCESSED_DIR: str = 'processed'
DATASETS_IMAGE_DIR: str = 'images'
DATASETS_LABEL_DIR: str = 'labels'
TRAIN_DATASET_FILE: str = 'train.hdf5'
VALIDATION_DATASET_FILE: str = 'validation.hdf5'

OUTPUT_ROOT_DIR: str = 'output'
OUTPUT_CHECKPOINTS_DIR: str = 'checkpoints'
OUTPUT_PREDICTIONS_DIR: str = 'predictions'
OUTPUT_WEIGHTS_DIR: str = 'weights'
OUTPUT_MODELS_DIR: str = 'models'
LOG_FILE: str = 'log.csv'
MODEL_FILE: str = 'model.json'
METRICS_FILE: str = 'metrics.json'
FIGURE_FILE: str = 'figure.png'

@dataclass(slots=True)
class Config:
    """Config dataclass, holding all hyperparameters as indicated by the user as well as some defaults."""
    id: str
    dataset_dir: str

    # Model info
    model: ModelType
    backbone: BackboneType
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

    # Dataset info
    image_dims: tuple[int, int, int]

    validation_split_percent: float
    binarize_labels: bool
    augment_data: bool

    dataset_images_dir: str # Inferred, uses dataset_dir
    dataset_labels_dir: str # Inferred, uses dataset_dir
    dataset_train_set_file: str # Inferred, uses dataset_dir
    dataset_validation_set_file: str # Inferred, uses dataset_dir

    # Output/logging info
    save_model: bool    # Whether to save the entire model (or if False, the weights)
    epochs_per_checkpoint: int

    output_checkpoints_dir: str # Inferred, uses id
    output_predictions_dir: str # Inferred, uses id
    output_weights_dir: str # Inferred, uses id
    output_models_dir: str # Inferred, uses id
    output_log_file: str # Inferred, uses id
    output_model_file: str # Inferred, uses id
    output_metrics_file: str # Inferred, uses id
    output_figure_file: str # Inferred, uses id

    # Prediction settings
    dilate_labels: bool
    prediction_file: Union[str, None]

    # Run-time constants, change at your own leisure
    START_EPOCH: int = 0
    MONITOR_METRIC: str = 'F1_score_dil'

    UNET_NUM_FILTERS: int = 64
    UNET_LAYER_WEIGHT_INITIALIZERS: UnetWeightInitializerType = UnetWeightInitializerType.HENormal

    FOCAL_LOSS_ALPHA: float = 0.25
    FOCAL_LOSS_GAMMA: float = 2.0
    WCE_BETA: float = 10

def load_config(filename: str) -> Config:
    """Load a config YAML file. Doesn't catch input errors that might be throw due to errors."""
    with open(filename, 'r') as config_file:
        config_vals = yaml.load(config_file, Loader=SafeLoader)
        return Config(
            id=str(config_vals['id']),
            dataset_dir=str(config_vals['dataset_dir']),
            model=ModelType(config_vals['model']),
            backbone=BackboneType(config_vals['backbone']),
            use_pretrained=bool(config_vals['use_pretrained']),
            batch_size=int(config_vals['batch_size']),
            batch_normalization=bool(config_vals['batch_normalization']),
            num_epochs=int(config_vals['num_epochs']),
            loss=LossType(config_vals['loss']),
            initial_learning_rate=float(config_vals['initial_learning_rate']),
            regularization=float(config_vals['regularization']),
            dropout=float(config_vals['dropout']) if config_vals.get('dropout') is not None else None,
            optimizer=OptimizerType(config_vals['optimizer']),
            image_dims=tuple(config_vals['image_dims']),
            validation_split_percent=float(config_vals['validation_split_percent']),
            binarize_labels=bool(config_vals['binarize_labels']),
            augment_data=bool(config_vals['augment_data']),
            dataset_images_dir=os.path.join(DATASETS_ROOT_DIR, config_vals['dataset_dir'], DATASETS_IMAGE_DIR),
            dataset_labels_dir=os.path.join(DATASETS_ROOT_DIR, config_vals['dataset_dir'], DATASETS_LABEL_DIR),
            dataset_train_set_file=os.path.join(DATASETS_ROOT_DIR, config_vals['dataset_dir'], TRAIN_DATASET_FILE),
            dataset_validation_set_file=os.path.join(DATASETS_ROOT_DIR, config_vals['dataset_dir'], VALIDATION_DATASET_FILE),
            save_model=bool(config_vals['save_model']),
            epochs_per_checkpoint=int(config_vals['epochs_per_checkpoint']),
            output_checkpoints_dir=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], OUTPUT_CHECKPOINTS_DIR),
            output_predictions_dir=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], OUTPUT_PREDICTIONS_DIR),
            output_weights_dir=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], OUTPUT_WEIGHTS_DIR),
            output_models_dir=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], OUTPUT_MODELS_DIR),
            output_log_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], LOG_FILE),
            output_model_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], MODEL_FILE),
            output_metrics_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], METRICS_FILE),
            output_figure_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], FIGURE_FILE),
            dilate_labels=bool(config_vals['dilate_labels']),
            prediction_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], OUTPUT_WEIGHTS_DIR, config_vals['prediction_file']) if config_vals.get('prediction_file') is not None else None
        )
