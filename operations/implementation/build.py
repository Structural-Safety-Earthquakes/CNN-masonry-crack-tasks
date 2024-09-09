import cv2
import numpy as np
import progressbar
from imutils import paths
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from operations.operation import Operation
import operations.arguments as arguments
from typing import Any
from subroutines.HDF5 import HDF5DatasetWriterMask
from util.dataset_config import load_data_config


class Build(Operation):
    """Build the config."""

    def __call__(self, dataset: str) -> None:
        """Process the dataset into a training and validation set."""
        config = load_data_config(dataset)

        # Grab the paths to the images and masks and sort them to ensure everything is properly in order.
        img_paths = list(paths.list_images(config.dataset_images_dir))
        label_paths = list(paths.list_images(config.dataset_labels_dir))
        img_paths.sort()
        label_paths.sort()

        if config.validation_split_percent == 0.:  # Only training data
            datasets = [(list(zip(img_paths, label_paths)), config.dataset_train_set_file)]
        elif config.validation_split_percent == 1.:  # Only validation data
            datasets = [(list(zip(img_paths, label_paths)), config.dataset_validation_set_file)]
        else:  # Perform stratified sampling from the training set to build the testing split from the training data
            train_img_split, val_img_split, train_label_split, val_label_split = train_test_split(
                img_paths,
                label_paths,
                test_size=config.validation_split_percent,
                random_state=42
            )

            # Construct a list pairing the training, validation, and testing image paths along with their corresponding labels
            # and output HDF5 files
            datasets = [
                (list(zip(train_img_split, train_label_split)), config.dataset_train_set_file),
                (list(zip(val_img_split, val_label_split)), config.dataset_validation_set_file)
            ]

        # Loop over the dataset tuples
        for (data_pairs, output_file) in datasets:
            print("[INFO] building {}...".format(output_file))

            # Create HDF5 writer
            writer = HDF5DatasetWriterMask((len(data_pairs), *config.image_dims), output_file)

            # initialize the progress bar
            widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
            pbar = progressbar.ProgressBar(maxval=len(data_pairs), widgets=widgets).start()

            # loop over the image paths
            for (idx, (im_path, label_path)) in enumerate(data_pairs):
                # Load the image and label and resize them if necessary
                image = cv2.imread(im_path)
                label = cv2.imread(label_path, 0)

                if config.image_dims != image.shape:
                    image = resize(image, config.image_dims, mode='constant', preserve_range=True)
                if config.image_dims[:2] != label.shape:
                    label = resize(label, config.image_dims[:2], mode='constant', preserve_range=True)

                # Normalize intensity values: [0,1]
                label = np.expand_dims(label, axis=-1)
                label = label / 255
                image = image / 255

                # Add to the HDF5 dataset and update the progress bar
                writer.add([image], [label])
                pbar.update(idx)

            # Close the progress bar and the HDF5 writer
            pbar.finish()
            writer.close()

    def get_cli_arguments(self) -> list[dict[str, Any]]:
        """We only need a dataset config."""
        return [
            arguments.DATASET_ARGUMENT
        ]
