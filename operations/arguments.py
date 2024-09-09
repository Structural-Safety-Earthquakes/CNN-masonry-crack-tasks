DATASET_ARGUMENT = {
    'name': ['--dataset', '-d'],
    'help': 'Dataset config file',
    'type': str,
    'required': True
}
NETWORK_ARGUMENT = {
    'name': ['--network', '-n'],
    'help': 'Network config file',
    'type': str,
    'required': True
}
WEIGHTS_FILE_ARGUMENT = {
    'name': ['--weights'],
    'help': 'Name of the file containing the model weights. Must either be located in the "weights" folder of the current dataset-network combo or should be the path to the file using the same network. Leave empty to use the highest scoring weights file.',
    'type': str,
    'default': None,
    'required': False
}
DILATE_VALIDATION_LABELS_ARGUMENT = {
    'name': ['--dilate'],
    'help': 'Whether to dilate the validation labels.',
    'type': str,    # We need to manually parse this due to how Python casts a str to bool
    'default': 'True',
    'required': False
}
