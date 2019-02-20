"""Logger module.

This module instantiates a global logger singleton.
"""
from garage.logger.logger import Logger
from garage.logger.outputs import CsvOutput, LogOutput, NullOutput, StdOutput
from garage.logger.outputs import TextOutput
from garage.logger.snapshotter import Snapshotter
from garage.logger.tabular_input import TabularInput
from garage.logger.tensorboard_inputs import HistogramInput
from garage.logger.tensorboard_inputs import HistogramInputDistribution
from garage.logger.tensorboard_inputs import HistogramInputGamma
from garage.logger.tensorboard_inputs import HistogramInputNormal
from garage.logger.tensorboard_inputs import HistogramInputPoisson
from garage.logger.tensorboard_inputs import HistogramInputUniform
from garage.logger.tensorboard_output import TensorBoardOutput

logger = Logger()
tabular = TabularInput()
snapshotter = Snapshotter()

__all__ = ("Logger", "NullOutput", "CsvOutput", "StdOutput", "TextOutput",
           "LogOutput", "Snapshotter", "TabularInput", "HistogramInput",
           "HistogramInputDistribution", "HistogramInputGamma",
           "HistogramInputNormal", "HistogramInputPoisson",
           "HistogramInputUniform", "TensorBoardOutput", "logger", "tabular",
           "snapshotter")
