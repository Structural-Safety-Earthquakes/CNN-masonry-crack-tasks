import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union


# Path constants, change at your own leisure
DATASETS_ROOT_DIR: str = 'dataset'
DATASETS_IMAGE_DIR: str = 'images'
DATASETS_LABEL_DIR: str = 'labels'

TRAIN_DATASET_FILE: str = 'train.hdf5'
VALIDATION_DATASET_FILE: str = 'validation.hdf5'

OUTPUT_ROOT_DIR: str = 'output'
OUTPUT_CHECKPOINTS_DIR: str = 'checkpoints'
OUTPUT_PREDICTIONS_DIR: str = 'predictions'
OUTPUT_BEST_MODELS_DIR: str = 'best_models'

LOG_FILE: str = 'log.csv'
MODEL_FILE: str = 'model.json'
METRICS_FILE: str = 'metrics.json'
PROGRESSION_FILE: str = 'progression.png'
NETWORK_FIGURE_FILE: str = 'network.png'
TXT_SUMMARY_FILE: str = 'summary.txt'

@dataclass(slots=True)
class OutputConfig:
    """Config specifying all output directories/files."""
    # Network specific (using network_id)
    network_dir: Union[str, None] = None

    # Dataset specific (using dataset_id)
    images_dir: Union[str, None] = None
    labels_dir: Union[str, None] = None
    train_set_file: Union[str, None] = None
    validation_set_file: Union[str, None] = None

    # Dataset & Network specific (using dataset_id and network_id)
    checkpoints_dir: Union[str, None] = None
    predictions_dir: Union[str, None] = None
    best_models_dir: Union[str, None] = None
    log_file: Union[str, None] = None
    model_file: Union[str, None] = None
    metrics_file: Union[str, None] = None
    progression_file: Union[str, None] = None
    figure_file: Union[str, None] = None
    txt_summary_file: Union[str, None] = None

def load_output_config(network_id: Union[str, None] = None, dataset_id: Union[str, None] = None) -> OutputConfig:
    """Load an output config using either the network_id, dataset_id or both."""
    output_config = OutputConfig()
    if network_id is not None:
        output_config.network_dir = os.path.join(OUTPUT_ROOT_DIR, network_id)

    if dataset_id is not None:
        dataset_dir = os.path.join(DATASETS_ROOT_DIR, dataset_id)

        output_config.images_dir = os.path.join(dataset_dir, DATASETS_IMAGE_DIR)
        output_config.labels_dir = os.path.join(dataset_dir, DATASETS_LABEL_DIR)
        output_config.train_set_file = os.path.join(dataset_dir, TRAIN_DATASET_FILE)
        output_config.validation_set_file = os.path.join(dataset_dir, VALIDATION_DATASET_FILE)

    if network_id is not None and dataset_id is not None:
        output_dir = os.path.join(OUTPUT_ROOT_DIR, network_id, dataset_id)
        Path(output_dir).parent.mkdir(exist_ok=True)
        Path(output_dir).mkdir(exist_ok=True)

        output_config.log_file = os.path.join(output_dir, LOG_FILE)
        output_config.model_file = os.path.join(output_dir, MODEL_FILE)
        output_config.metrics_file = os.path.join(output_dir, METRICS_FILE)
        output_config.progression_file = os.path.join(output_dir, PROGRESSION_FILE)
        output_config.figure_file = os.path.join(output_dir, NETWORK_FIGURE_FILE)
        output_config.txt_summary_file = os.path.join(output_dir, TXT_SUMMARY_FILE)

        output_config.checkpoints_dir = os.path.join(output_dir, OUTPUT_CHECKPOINTS_DIR)
        output_config.predictions_dir = os.path.join(output_dir, OUTPUT_PREDICTIONS_DIR)
        output_config.best_models_dir = os.path.join(output_dir, OUTPUT_BEST_MODELS_DIR)
        Path(output_config.checkpoints_dir).mkdir(exist_ok=True)
        Path(output_config.predictions_dir).mkdir(exist_ok=True)
        Path(output_config.best_models_dir).mkdir(exist_ok=True)

    return output_config
