import tempfile
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tensorflow as tf

from garage.logger import Histogram
from garage.logger import TabularInput
from garage.logger import TensorBoardOutput
from tests.fixtures import TfGraphTestCase


class TBOutputTest(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.log_dir = tempfile.TemporaryDirectory()
        self.tabular = TabularInput()
        self.tabular.clear()
        self.tensor_board_output = TensorBoardOutput(self.log_dir.name)

    def tearDown(self):
        self.tensor_board_output.close()
        self.log_dir.cleanup()
        super().tearDown()


class TestTensorBoardOutput(TBOutputTest):
    """ Simple test without mocks.

    This makes sure that real file output is still functioning.
    """

    def test_record_scalar(self):
        foo = 5
        bar = 10.0
        self.tabular.record('foo', foo)
        self.tabular.record('bar', bar)
        self.tensor_board_output.record(self.tabular)
        self.tensor_board_output.dump()


class TestTensorBoardOutputMocked(TBOutputTest):
    """ Full test which mocks out TensorboardX. """

    def run(self, result=None):
        with mock.patch('tensorboardX.SummaryWriter'):
            super().run(result)

    def setUp(self):
        super().setUp()
        self.mock_writer = self.tensor_board_output._writer

    def test_record_graph(self):
        foo = tf.constant(5)  # noqa: F841
        self.tensor_board_output.record(self.graph)

        self.mock_writer.file_writer.add_event.assert_called_once()

    def test_record_scalar(self):
        foo = 5
        bar = 10.0
        self.tabular.record('foo', foo)
        self.tabular.record('bar', bar)
        self.tensor_board_output.record(self.tabular)
        self.tensor_board_output.dump()

        self.mock_writer.add_scalar.assert_any_call('foo', foo, 0)
        self.mock_writer.add_scalar.assert_any_call('bar', bar, 0)

    def test_record_figure(self):
        fig = plt.figure()
        ax = fig.gca()
        xs = np.arange(10.0)
        ys = xs**2
        ax.scatter(xs, ys)
        self.tabular.record('baz', fig)
        self.tensor_board_output.record(self.tabular)
        self.tensor_board_output.dump()

        self.mock_writer.add_figure.assert_called_once_with('baz', fig, 0)

    def test_record_tabular(self):
        foo = 5
        bar = 10.0
        self.tabular.record('foo', foo)
        self.tabular.record('bar', bar)
        self.tensor_board_output.record(self.tabular, prefix='a/')
        self.tensor_board_output.dump()

        self.mock_writer.add_scalar.assert_any_call('foo', foo, 0)
        self.mock_writer.add_scalar.assert_any_call('bar', bar, 0)

    def test_record_scipy_stats_distribution(self):
        shape = np.ones((2, 10))
        normal = scipy.stats.norm(loc=0.1 * shape, scale=2.0 * shape)
        gamma = scipy.stats.gamma(a=0.2 * shape)
        poisson = scipy.stats.poisson(mu=0.3 * shape)
        uniform = scipy.stats.randint(high=shape, low=-shape)

        self.tabular.record('Normal', normal)
        self.tabular.record('Gamma', gamma)
        self.tabular.record('Poisson', poisson)
        self.tabular.record('Uniform', uniform)
        self.tensor_board_output.record(self.tabular)
        self.tensor_board_output.dump()

        assert self.mock_writer.add_histogram.call_count == 4

    def test_record_scipy_stats_multivariate_distribution(self):
        mvn = scipy.stats.multivariate_normal(
            mean=np.ones(10), cov=2.0 * np.ones(10))

        self.tabular.record('MultivariateNormal', mvn)
        self.tensor_board_output.record(self.tabular)
        self.tensor_board_output.dump()

        assert self.mock_writer.add_histogram.call_count == 1

    def test_record_histogram(self):
        norm = scipy.stats.norm(loc=[1., 0.], scale=[0.5, 1.5])
        samples = norm.rvs((10000, 2))
        hist = Histogram(samples)
        self.tabular.record('Samples', hist)
        self.tensor_board_output.record(self.tabular)
        self.tensor_board_output.dump()

        assert self.mock_writer.add_histogram.call_count == 1

    def test_unknown_tabular_value(self):
        self.tabular.record('foo', dict())
        self.tensor_board_output.record(self.tabular)
        self.tensor_board_output.dump()
        # 'foo' should be silently ignored

    def test_unknown_input_type(self):
        with self.assertRaises(ValueError):
            foo = np.zeros((3, 10))
            self.tensor_board_output.record(foo)
