"""Global logger module.

This module instantiates a global logger according to configuration.
Set the environment varialbe GARAGE_LOG_TENSORBOARD to enable TensorBoard.
"""
import garage.config as config

if getattr(config, "GARAGE_LOG_TENSORBOARD", False):
    from garage.misc.logger.tensorboard_logger import TensorboardLogger
    logger = TensorboardLogger()
else:
    from garage.misc.logger.base_logger import Logger
    logger = Logger()

__all__ = ["logger"]
