import os

from network.loss import determine_loss_function
from network.metrics import get_standard_metrics
from network_class import Network
from optimizer_class import Optimizer
from subroutines.callbacks import EpochCheckpoint, TrainingMonitor
from subroutines.HDF5 import HDF5DatasetGeneratorMask
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from util.config import Config


def train_model(config: Config):
    """Train a model given the specific config."""

    # TODO: remove args by refactoring dependencies
    args = {
        'main': os.getcwd(),
        'opt': config.optimizer.value,
        'regularization': config.regularization,
        'model': f'sm_{config.model.value}_{config.backbone.value}' if config.backbone is not None else config.model.value,
        'dropout': config.dropout,
        'batchnorm': config.batch_normalization,
        'init': config.UNET_LAYER_WEIGHT_INITIALIZERS.value,
        'encoder_weights': 'imagenet' if config.use_pretrained else None
    }

    #%%
    # Prepare model for training
    #
    model = Network(
        args,
        config.dataset_config.image_dims,
        config.UNET_NUM_FILTERS,
        config.batch_size,
        config.initial_learning_rate,
        Optimizer(args, config.initial_learning_rate).define_Optimizer(),
        determine_loss_function(config),
        get_standard_metrics()
    ).define_Network()

    # Data augmentation for training and validation sets
    if config.augment_data:
        aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
            width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
            horizontal_flip=True, fill_mode='nearest')
    else:
        aug = None

    # Load data generators
    train_gen = HDF5DatasetGeneratorMask(
        config.dataset_config.dataset_train_set_file,
        config.batch_size,
        aug=aug,
        shuffle=False,
        binarize=config.binarize_labels
    )
    val_gen = HDF5DatasetGeneratorMask(
        config.dataset_config.dataset_validation_set_file,
        config.batch_size,
        aug=aug,
        shuffle=False,
        binarize=config.binarize_labels
    )

    #%%

    # Callback that streams epoch results to a CSV file
    # https://keras.io/api/callbacks/csv_logger/
    csv_logger = CSVLogger(config.output_log_file, append=True, separator=';')

    # Serialize model to JSON
    try:
        model_json = model.to_json()
        with open(config.output_model_file, 'w') as json_file:
            json_file.write(model_json)
    except:
        print('Warning: Unable to write model.json!!')

    # Define whether the whole model or the weights only will be saved from the ModelCheckpoint
    # Refer to the documentation of ModelCheckpoint for extra details
    # https://keras.io/api/callbacks/model_checkpoint/
    template_name = f'epoch_{{epoch}}_{config.MONITOR_METRIC}_{{val_{config.MONITOR_METRIC}:.3f}}.h5'
    checkpoint_file = os.path.join(config.output_checkpoints_dir, template_name) if config.save_model else os.path.join(config.output_weights_dir, template_name)

    epoch_checkpoint = EpochCheckpoint(
        config.output_checkpoints_dir,
        config.output_weights_dir,
        'model' if config.save_model else 'weights',
        every=config.epochs_per_checkpoint,
        startAt=config.START_EPOCH,
        info=config.id,
        counter='0'
    )
    training_monitor = TrainingMonitor(
        config.output_figure_file,
        jsonPath=config.output_metrics_file,
        startAt=config.START_EPOCH,
        metric=config.MONITOR_METRIC
    )
    model_checkpoint = ModelCheckpoint(
        checkpoint_file,
        monitor=f'val_{config.MONITOR_METRIC}',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=not config.save_model
    )

    #%%
    # Train the network
    #
    H = model.fit(
        train_gen.generator(),
        steps_per_epoch=train_gen.numImages // config.batch_size,
        validation_data=val_gen.generator(),
        validation_steps=val_gen.numImages // config.batch_size,
        epochs=config.num_epochs,
        max_queue_size=config.batch_size * 2,
        callbacks=[
            csv_logger,
            epoch_checkpoint,
            training_monitor,
            model_checkpoint
        ],
        verbose=1
    )