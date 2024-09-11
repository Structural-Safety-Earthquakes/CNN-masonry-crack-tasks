import h5py

from util.hdf5 import IMAGES_KEY, LABELS_KEY


class HDF5DatasetWriter:
    """File writer for writing images and labels to a HDF5 file."""

    BUFFER_SIZE = 1000 # Size of the write buffer

    data_file: h5py.File
    images_dataset: h5py.Dataset
    labels_dataset: h5py.Dataset

    dataset_cursor: int
    buffer: dict[str, list]

    def __init__(self, image_dims: tuple[int, int, int, int], output_path: str):
        self.data_file = h5py.File(output_path, 'w')
        self.images_dataset = self.data_file.create_dataset(IMAGES_KEY, image_dims, dtype=float)
        self.labels_dataset = self.data_file.create_dataset(LABELS_KEY, image_dims[:-1] + (1,), dtype=float)

        self.dataset_cursor = 0
        self.buffer = {IMAGES_KEY: [], LABELS_KEY: []}

    def add(self, images: list, labels: list) -> None:
        """Add data to the buffer and flush if needed."""
        self.buffer[IMAGES_KEY] += images
        self.buffer[LABELS_KEY] += labels

        if len(self.buffer[IMAGES_KEY]) >= self.BUFFER_SIZE:
            self.flush()
            
    def flush(self):
        """Flush to disk and reset the buffer."""
        end_cursor = self.dataset_cursor + len(self.buffer[IMAGES_KEY])
        self.images_dataset[self.dataset_cursor:end_cursor] = self.buffer[IMAGES_KEY]
        self.labels_dataset[self.dataset_cursor:end_cursor] = self.buffer[LABELS_KEY]

        self.dataset_cursor = end_cursor
        self.buffer[IMAGES_KEY].clear()
        self.buffer[LABELS_KEY].clear()
            
    def close(self):
        """Close the file, flush first if needed."""
        if len(self.buffer[IMAGES_KEY]) > 0:
            self.flush()

        self.data_file.close()
