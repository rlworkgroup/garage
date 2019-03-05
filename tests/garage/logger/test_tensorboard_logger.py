import random
import unittest

import numpy as np
import tensorflow as tf

from garage.logger import HistogramInput, HistogramInputGamma
from garage.logger import HistogramInputNormal, HistogramInputPoisson
from garage.logger import HistogramInputUniform, TensorBoardOutput
from garage.logger import logger, tabular
from garage.misc.console import remove_dir_if_exists


class TestTensorBoardLogger(unittest.TestCase):
    def test_histogram(self):
        log_dir = 'test'
        remove_dir_if_exists(log_dir)
        try:
            tb_output = TensorBoardOutput(log_dir)
            logger.add_output(tb_output)

            foo = random.randint(0, 999)
            bar = random.randint(0, 999)
            baz = np.random.rand(10).tolist()
            tabular.record("foo", foo)
            tabular.record("bar", bar)
            tabular.record("hi", HistogramInput(baz))
            with tabular.prefix('hid\\'):
                tabular.record(
                    "normal",
                    HistogramInputNormal(
                        shape=[1000, 10],
                        mean=np.random.rand(1),
                        stddev=np.random.rand(1)))
                tabular.record(
                    "gamma",
                    HistogramInputGamma(
                        shape=[1000, 10], alpha=np.random.rand(1)))
                tabular.record(
                    "poisson",
                    HistogramInputPoisson(
                        shape=[1000, 10], lam=np.random.rand(1)))
                tabular.record(
                    "uniform",
                    HistogramInputUniform(
                        shape=[1000, 10], maxval=np.random.rand(1)))

            tb_output.dump()  # this should warn 'Cannot dump histogram.'

            sess = tf.Session()
            sess.__enter__()

            logger.log(tabular)
            tb_output.dump()

        finally:
            logger.remove_all()

            # comment this line to check output in tensorboard
            remove_dir_if_exists(log_dir)
