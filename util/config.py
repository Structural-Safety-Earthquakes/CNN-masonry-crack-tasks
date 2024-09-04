import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml
from yaml import SafeLoader

from util.dataset_config import DatasetConfig, load_data_config
from util.types import OptimizerType, ModelType, BackboneType, UnetWeightInitializerType, LossType

# Path constants, change at your own leisure
OUTPUT_ROOT_DIR: str = 'output'
OUTPUT_CHECKPOINTS_DIR: str = 'checkpoints'
OUTPUT_PREDICTIONS_DIR: str = 'predictions'
OUTPUT_WEIGHTS_DIR: str = 'weights'
OUTPUT_MODELS_DIR: str = 'models'
LOG_FILE: str = 'log.csv'
MODEL_FILE: str = 'model.json'
METRICS_FILE: str = 'metrics.json'
FIGURE_FILE: str = 'figure.png'
NETWORK_FIGURE_FILE: str = 'network.png'
TXT_SUMMARY_FILE: str = 'summary.txt'

@dataclass(slots=True)
class Config:
    """Config dataclass, holding most hyperparameters as indicated by the user as well as some defaults."""
    id: str

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

    augment_data: bool
    binarize_labels: bool

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
    output_network_figure_file: str # Inferred, uses id
    output_txt_summary_file: str  # Inferred, uses id

    # Prediction settings
    dilate_labels: bool
    prediction_file: Union[str, None]   # Takes the highest trained model in case of None

    # Nested configs
    dataset_config: DatasetConfig

    # Run-time constants, change at your own leisure
    START_EPOCH: int = 0
    MONITOR_METRIC: str = 'f1_score_dilated'

    UNET_NUM_FILTERS: int = 64
    UNET_LAYER_WEIGHT_INITIALIZERS: UnetWeightInitializerType = UnetWeightInitializerType.HENormal

    FOCAL_LOSS_ALPHA: float = 0.25
    FOCAL_LOSS_GAMMA: float = 2.0
    WCE_BETA: float = 10

def load_config(config_filename: str, dataset_config_filename) -> Config:
    """Load a config YAML file. Doesn't catch input errors that might be throw due to errors."""

    dataset_config = load_data_config(dataset_config_filename)

    with open(config_filename, 'r') as config_file:
        config_vals = yaml.load(config_file, Loader=SafeLoader)

    # Try loading the config
    config = Config(
        id=str(config_vals['id']),
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
        augment_data=bool(config_vals['augment_data']),
        binarize_labels=bool(config_vals['binarize_labels']),
        save_model=bool(config_vals['save_model']),
        epochs_per_checkpoint=int(config_vals['epochs_per_checkpoint']),
        output_checkpoints_dir=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], dataset_config.dataset_dir, OUTPUT_CHECKPOINTS_DIR),
        output_predictions_dir=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], dataset_config.dataset_dir, OUTPUT_PREDICTIONS_DIR),
        output_weights_dir=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], dataset_config.dataset_dir, OUTPUT_WEIGHTS_DIR),
        output_models_dir=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], dataset_config.dataset_dir, OUTPUT_MODELS_DIR),
        output_log_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], dataset_config.dataset_dir, LOG_FILE),
        output_model_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], dataset_config.dataset_dir, MODEL_FILE),
        output_metrics_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], dataset_config.dataset_dir, METRICS_FILE),
        output_figure_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], dataset_config.dataset_dir, FIGURE_FILE),
        output_network_figure_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], NETWORK_FIGURE_FILE),
        output_txt_summary_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], TXT_SUMMARY_FILE),
        dilate_labels=bool(config_vals['dilate_labels']),
        prediction_file=os.path.join(OUTPUT_ROOT_DIR, config_vals['id'], OUTPUT_WEIGHTS_DIR, config_vals['prediction_file']) if config_vals.get('prediction_file') is not None else None,
        dataset_config=dataset_config
    )

    # Create dirs that don't exist
    Path(config.output_checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_predictions_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_weights_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_models_dir).mkdir(parents=True, exist_ok=True)

    return config
