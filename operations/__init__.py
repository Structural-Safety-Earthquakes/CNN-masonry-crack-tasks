from operations.implementation.build import Build
from operations.implementation.test import Test
from operations.implementation.train import Train
from operations.implementation.visualize import Visualize
from operations.operation import Operation
from util.types import OperationType


def get_operation(operation_type: OperationType) -> Operation:
    """Get the operation associated with the type."""
    match operation_type:
        case OperationType.Build:
            return Build()
        case OperationType.Train:
            return Train()
        case OperationType.Test:
            return Test()
        case OperationType.Visualize:
            return Visualize()
        case _:
            raise ValueError(f'Unknown operation: {operation_type}')