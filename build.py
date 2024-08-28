import cv2
import numpy as np
import progressbar
from imutils import paths
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from subroutines.HDF5 import HDF5DatasetWriterMask
from util.config import Config


def process_dataset(config: Config):
    """Process the dataset into a training and validation set."""

    # Grab the paths to the images and masks and sort them to ensure everything is properly in order.
    img_paths = list(paths.list_images(config.dataset_images_dir))
    label_paths = list(paths.list_images(config.dataset_labels_dir))
    img_paths.sort()
    label_paths.sort()

    # Perform stratified sampling from the training set to build the testing split from the training data
    train_img_split, val_img_split, train_label_split, val_label_split = train_test_split(
        img_paths,
        label_paths,
        test_size=config.validation_split_percent,
        random_state=42
    )

    # Construct a list pairing the training, validation, and testing image paths along with their corresponding labels
    # and output HDF5 files
    datasets = [
        (zip(train_img_split, train_label_split), config.dataset_train_set_file),
        (zip(val_img_split, val_label_split), config.dataset_validation_set_file)
    ]

    # Loop over the dataset tuples
    for (data_pairs, output_file) in datasets:
        print("[INFO] building {}...".format(output_file))

        # Create HDF5 writer
        writer = HDF5DatasetWriterMask([len(data_pairs), *config.image_dims], output_file)

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
