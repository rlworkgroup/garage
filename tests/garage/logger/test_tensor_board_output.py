import tempfile
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from garage.logger import TabularInput
from garage.logger import TensorBoardOutput
from tests.fixtures import TfTestCase


class TBOutputTest(TfTestCase):
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

    def test_graph_saved_only_once(self):
        self.tabular.record('foo', 1.0)
        self.tensor_board_output.dump()
        self.tabular.record('foo', 2.0)
        self.tensor_board_output.dump()

        self.mock_writer.add_graph.assert_called_once()

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

        self.mock_writer.add_scalar.assert_any_call('a/foo', foo, 0)
        self.mock_writer.add_scalar.assert_any_call('a/bar', bar, 0)

    def test_record_tfp_distribution(self):
        with tf.Session():
            histo_shape = np.ones((1000, 10))
            normal = tfp.distributions.Normal(
                loc=0.1 * histo_shape, scale=2.0 * histo_shape)
            gamma = tfp.distributions.Gamma(
                concentration=0.2 * histo_shape, rate=4.0 * histo_shape)
            poisson = tfp.distributions.Poisson(rate=0.3 * histo_shape)
            uniform = tfp.distributions.Uniform(high=1.0 * histo_shape)

            self.tabular.record('Normal', normal)
            self.tabular.record('Gamma', gamma)
            self.tabular.record('Poisson', poisson)
            self.tabular.record('Uniform', uniform)
            self.tensor_board_output.record(self.tabular)
            self.tensor_board_output.dump()

            assert self.mock_writer.add_histogram.call_count == 4

    def test_unknown_tabular_value(self):
        self.tabular.record('foo', dict())
        self.tensor_board_output.record(self.tabular)
        self.tensor_board_output.dump()
        # 'foo' should be silently ignored

    def test_unknown_input_type(self):
        with self.assertRaises(ValueError):
            foo = np.zeros((3, 10))
            self.tensor_board_output.record(foo)
