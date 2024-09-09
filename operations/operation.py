import abc
from abc import ABC
from typing import Any


class Operation(ABC):
    """
    Functor class which represents an operation.
    Requires a network config and/or a dataset config and optionally custom options.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        """Execute the operation."""
        pass

    @abc.abstractmethod
    def get_cli_arguments(self) -> list[dict[str, Any]]:
        """
        Get the argparse arguments associated with this operation.
        The name key can be used to clarify arguments that should be used as the CLI argument name or flag.
        """
        return []
