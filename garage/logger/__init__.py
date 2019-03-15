"""Logger module.

This module instantiates a global logger singleton.
"""
from garage.logger.logger import Logger, LogOutput
from garage.logger.simple_outputs import NullOutput, StdOutput, TextOutput
from garage.logger.tabular_input import TabularInput
from garage.logger.csv_output import CsvOutput  # noqa: I100
from garage.logger.snapshotter import Snapshotter
from garage.logger.tensor_board_output import TensorBoardOutput

logger = Logger()
tabular = TabularInput()
snapshotter = Snapshotter()

__all__ = ('Logger', 'NullOutput', 'CsvOutput', 'StdOutput', 'TextOutput',
           'LogOutput', 'Snapshotter', 'TabularInput', 'TensorBoardOutput',
           'logger', 'tabular', 'snapshotter')
