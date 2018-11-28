import garage.config as config

if getattr(config, "LOG_TENSORBOARD", False):
    from garage.misc.logger.tensorboard_logger import TensorboardLogger
    logger = TensorboardLogger()
else:
    from garage.misc.logger.base_logger import Logger
    logger = Logger()

__all__ = ["logger"]
