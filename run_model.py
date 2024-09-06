import os

os.environ['TF_USE_LEGACY_KERAS'] = '1' # Fall back to Keras 2. We need this before loading **anything** else
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import argparse

from build import process_dataset
from test import generate_predictions
from train import train_model
from visualize import visualize_architecture
from util.config import load_config
from util.types import RunMode


def run_model():
    """Entrypoint of the program"""
    # Gather user input
    parser = argparse.ArgumentParser(
        prog='CNN-masonry-crack-tasks',
        description='Train or evaluate crack segmentation models',
    )
    parser.add_argument('--config', '-c', help='Config file', type=str, required=True)
    parser.add_argument('--mode', '-m', help='Train, test, build or visualize', type=RunMode, required=True)
    parser.add_argument('--dataset', '-d', help='Dataset config file', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config, args.dataset)

    # Run actual program depending on mode
    if args.mode == RunMode.BUILD:
        process_dataset(config.dataset_config)
    if args.mode == RunMode.TRAIN:
        train_model(config)
    if args.mode == RunMode.TEST:
        generate_predictions(config)
    if args.mode == RunMode.VISUALIZE:
        visualize_architecture(config)

if __name__ == '__main__':
    run_model()