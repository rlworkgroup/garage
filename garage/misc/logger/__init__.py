"""Global logger module.

This module instantiates a global logger according to configuration.
Use LOG_TENSORBOARD=True in config.py to enable Tensorboard.
"""
import garage.config as config

if getattr(config, "LOG_TENSORBOARD", False):
    from garage.misc.logger.tensorboard_logger import TensorboardLogger
    logger = TensorboardLogger()
else:
    from garage.misc.logger.base_logger import Logger
    logger = Logger()

__all__ = ["logger"]
