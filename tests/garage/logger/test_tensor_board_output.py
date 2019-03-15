import random

import numpy as np

from garage.logger import distributions, logger, tabular, TensorBoardOutput
from garage.misc.console import remove_dir_if_exists
from tests.fixtures import TfTestCase


class TestTensorBoardOutput(TfTestCase):
    def test_histogram(self):
        log_dir = 'test'
        remove_dir_if_exists(log_dir)
        try:
            logger.remove_all()
            tb_output = TensorBoardOutput(log_dir)
            logger.add_output(tb_output)
            tabular.clear()

            foo = random.randint(0, 998)
            bar = random.randint(0, 998)
            baz = np.random.rand(10).tolist()
            tabular.record('foo', foo)
            tabular.record('bar', bar)
            tabular.record('hi', distributions.Empirical(baz))
            with tabular.prefix('hid\\'):
                tabular.record(
                    'normal',
                    distributions.Normal(
                        shape=[1000, 10],
                        mean=np.random.rand(1),
                        stddev=np.random.rand(1)))
                tabular.record(
                    'gamma',
                    distributions.Gamma(
                        shape=[1000, 10], alpha=np.random.rand(1)))
                tabular.record(
                    'poisson',
                    distributions.Poisson(
                        shape=[1000, 10], lam=np.random.rand(1)))
                tabular.record(
                    'uniform',
                    distributions.Uniform(
                        shape=[1000, 10], maxval=np.random.rand(1)))

            tb_output.dump()  # this should warn 'Cannot dump histogram.'

            logger.log(tabular)
            tb_output.dump()

            tabular.clear()
            tabular.record('foo', foo + 1)
            tabular.record('bar', bar + 1)
            logger.log(tabular)
            logger.dump_output_type(TensorBoardOutput)

        finally:
            logger.remove_all()

            # comment this line to check output in tensorboard
            remove_dir_if_exists(log_dir)
