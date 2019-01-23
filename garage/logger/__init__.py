"""Logger module.

This module instantiates a global logger singleton.
"""
from garage.logger.logger import Logger
from garage.logger.outputs import CsvOutput, StdOutput, TextOutput
from garage.logger.snapshotter import Snapshotter
from garage.logger.tabular_input import TabularInput
from garage.logger.tensorboard_output import TensorBoardOutput

logger = Logger()
tabular = TabularInput()
snapshotter = Snapshotter()

__all__ = [
    "tabular", "logger", "snapshotter", "CsvOutput", "StdOutput", "TextOutput",
    "TensorBoardOutput"
]
