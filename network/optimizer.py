import platform

from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.src.optimizers import optimizer

from util.config import Config
from util.types import OptimizerType


def determine_optimizer(config: Config) -> optimizer.Optimizer:
    """Determine the optimizer from the config. For Adam it is recommended to run the legacy version on Apple silicon."""
    match config.optimizer:
        case OptimizerType.Adam:
            is_apple_silicon = platform.system() == 'Darwin' and platform.processor() == 'arm'
            return LegacyAdam(config.initial_learning_rate) if is_apple_silicon else Adam(config.initial_learning_rate)
        case OptimizerType.SGD:
            return SGD(config.initial_learning_rate)
        case OptimizerType.RMSprop:
            return RMSprop(config.initial_learning_rate)
        case _:
            raise ValueError(f'Unknown loss type: {config.loss}')
