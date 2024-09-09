from operations.implementation.build import Build
from operations.implementation.test import Test
from operations.implementation.train import Train
from operations.implementation.visualize import Visualize
from operations.operation import Operation
from util.types import OperationType


def get_operation(operation_type: OperationType) -> Operation:
    """Get the operation associated with the type."""
    match operation_type:
        case OperationType.BUILD:
            return Build()
        case OperationType.TRAIN:
            return Train()
        case OperationType.TEST:
            return Test()
        case OperationType.VISUALIZE:
            return Visualize()
        case _:
            raise ValueError(f'Unknown operation: {operation_type}')