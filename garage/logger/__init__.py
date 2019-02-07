"""Logger module.

This module instantiates a global logger singleton.
"""
from garage.logger.logger import Logger
from garage.logger.outputs import CsvOutput, StdOutput, TextOutput
from garage.logger.snapshotter import Snapshotter
from garage.logger.tabular_input import TabularInput
from garage.logger.tensorboard_inputs import (
    HistogramInput, HistogramInputDistribution, HistogramInputGamma,
    HistogramInputNormal, HistogramInputPoisson, HistogramInputUniform)
from garage.logger.tensorboard_output import TensorBoardOutput

logger = Logger()
tabular = TabularInput()
snapshotter = Snapshotter()

__all__ = [
    "tabular", "logger", "HistogramInput", "HistogramInputDistribution",
    "HistogramInputNormal", "HistogramInputGamma", "HistogramInputPoisson",
    "HistogramInputUniform", "CsvOutput", "StdOutput", "TextOutput",
    "TensorBoardOutput", "snapshotter"
]
