import os
from typing import Any

from network.loss import determine_loss_function
from network.metrics import get_standard_metrics
from network.model import build_model
from network.optimizer import determine_optimizer
from operations.operation import Operation
import operations.arguments as arguments
from subroutines.callbacks import EpochCheckpoint, TrainingMonitor
from subroutines.HDF5 import HDF5DatasetGeneratorMask
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from util.config import load_network_config, load_data_config


class Train(Operation):
    """Train a model on a dataset."""

    def __call__(self, dataset: str, network: str) -> None:
        """Start training a model on a dataset."""
        network_config = load_network_config(network, dataset)
        dataset_config = load_data_config(dataset)

        # %%
        # Prepare model for training
        #
        model = build_model(network_config)
        model.compile(
            optimizer=determine_optimizer(network_config),
            loss=determine_loss_function(network_config),
            metrics=[get_standard_metrics()]
        )

        # Data augmentation for training and validation sets
        if network_config.augment_data:
            aug = ImageDataGenerator(
                rotation_range=20,
                zoom_range=0.15,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            aug = None

        # Load data generators
        train_gen = HDF5DatasetGeneratorMask(
            dataset_config.dataset_train_set_file,
            network_config.batch_size,
            aug=aug,
            shuffle=False,
            binarize=network_config.binarize_labels
        )
        val_gen = HDF5DatasetGeneratorMask(
            network_config.dataset_config.dataset_validation_set_file,
            network_config.batch_size,
            aug=aug,
            shuffle=False,
            binarize=network_config.binarize_labels
        )

        # %%

        # Callback that streams epoch results to a CSV file
        # https://keras.io/api/callbacks/csv_logger/
        csv_logger = CSVLogger(network_config.output_log_file, append=True, separator=';')

        # Serialize model to JSON
        try:
            model_json = model.to_json()
            with open(network_config.output_model_file, 'w') as json_file:
                json_file.write(model_json)
        except:
            print('Warning: Unable to write model.json!!')

        # Define whether the whole model or the weights only will be saved from the ModelCheckpoint
        # Refer to the documentation of ModelCheckpoint for extra details
        # https://keras.io/api/callbacks/model_checkpoint/
        template_name = f'epoch_{{epoch}}_{network_config.MONITOR_METRIC}_{{val_{network_config.MONITOR_METRIC}:.3f}}.h5'
        checkpoint_file = os.path.join(network_config.output_checkpoints_dir, template_name) if network_config.save_model else os.path.join(network_config.output_weights_dir, template_name)

        epoch_checkpoint = EpochCheckpoint(
            network_config.output_checkpoints_dir,
            network_config.output_weights_dir,
            'model' if network_config.save_model else 'weights',
            every=network_config.epochs_per_checkpoint,
            startAt=network_config.START_EPOCH,
            info=network_config.id,
            counter='0'
        )
        training_monitor = TrainingMonitor(
            network_config.output_figure_file,
            jsonPath=network_config.output_metrics_file,
            startAt=network_config.START_EPOCH,
            metric=network_config.MONITOR_METRIC
        )
        model_checkpoint = ModelCheckpoint(
            checkpoint_file,
            monitor=f'val_{network_config.MONITOR_METRIC}',
            verbose=1,
            save_best_only=True,
            mode='max',
            save_weights_only=not network_config.save_model
        )

        # %%
        # Train the network
        #
        H = model.fit(
            train_gen.generator(),
            steps_per_epoch=train_gen.numImages // network_config.batch_size,
            validation_data=val_gen.generator(),
            validation_steps=val_gen.numImages // network_config.batch_size,
            epochs=network_config.num_epochs,
            max_queue_size=network_config.batch_size * 2,
            callbacks=[
                csv_logger,
                epoch_checkpoint,
                training_monitor,
                model_checkpoint
            ],
            verbose=1
        )

    def get_cli_arguments(self) -> list[dict[str, Any]]:
        return [
            arguments.DATASET_ARGUMENT,
            arguments.NETWORK_ARGUMENT
        ]
