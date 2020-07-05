import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.models import GaussianCNNModel

from tests.fixtures import TfGraphTestCase


class TestGaussianCNNModel(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.batch_size = 5
        self.input_width = 10
        self.input_height = 10
        self.obs = np.full(
            (self.batch_size, self.input_width, self.input_height, 3), 1)
        input_shape = self.obs.shape[1:]  # height, width, channel
        self._input_ph = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, ) + input_shape,
                                                  name='input')

    @mock.patch('tensorflow.random.normal')
    @pytest.mark.parametrize('filters, in_channels, strides, output_dim, '
                             'hidden_sizes',
                             [(((3, (1, 1)), ), (3, ), (1, ), 1, (1, )),
                              (((3, (3, 3)), ), (3, ), (1, ), 2, (2, )),
                              (((3, (3, 3)), ), (3, ), (2, ), 1, (1, 1)),
                              (((3, (1, 1)), (3, (1, 1))), (3, 3), (1, 1), 2,
                               (2, 2)),
                              (((3, (3, 3)), (3, (3, 3))), (3, 3), (2, 2), 2,
                               (2, 2))])
    def test_std_share_network_output_values(self, mock_normal, filters,
                                             in_channels, strides, output_dim,
                                             hidden_sizes):
        mock_normal.return_value = 0.5
        model = GaussianCNNModel(filters=filters,
                                 strides=strides,
                                 padding='VALID',
                                 output_dim=output_dim,
                                 hidden_sizes=hidden_sizes,
                                 std_share_network=True,
                                 hidden_nonlinearity=None,
                                 std_parameterization='exp',
                                 hidden_w_init=tf.constant_initializer(0.01),
                                 output_w_init=tf.constant_initializer(1))
        outputs = model.build(self._input_ph).outputs

        action, mean, log_std, std_param = self.sess.run(
            outputs[:-1], feed_dict={self._input_ph: self.obs})

        filter_sum = 1

        for filter_iter, in_channel in zip(filters, in_channels):
            filter_height = filter_iter[1][0]
            filter_width = filter_iter[1][1]
            filter_sum *= 0.01 * filter_height * filter_width * in_channel

        for _ in hidden_sizes:
            filter_sum *= 0.01

        height_size = self.input_height
        width_size = self.input_width
        for filter_iter, stride in zip(filters, strides):
            height_size = int((height_size - filter_iter[1][0]) / stride) + 1
            width_size = int((width_size - filter_iter[1][1]) / stride) + 1
        flatten_shape = height_size * width_size * filters[-1][0]

        network_output = filter_sum * flatten_shape * np.prod(hidden_sizes)
        expected_mean = np.full((self.batch_size, output_dim),
                                network_output,
                                dtype=np.float32)
        expected_std_param = np.full((self.batch_size, output_dim),
                                     network_output,
                                     dtype=np.float32)
        expected_log_std = np.full((self.batch_size, output_dim),
                                   network_output,
                                   dtype=np.float32)

        assert np.allclose(mean, expected_mean)

        assert np.allclose(std_param, expected_std_param)
        assert np.allclose(log_std, expected_log_std)

        expected_action = 0.5 * np.exp(expected_log_std) + expected_mean
        assert np.allclose(action, expected_action, rtol=0, atol=0.1)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_std_share_network_shapes(self, output_dim, hidden_sizes):
        # should be 2 * output_dim
        model = GaussianCNNModel(filters=((3, (3, 3)), (6, (3, 3))),
                                 strides=[1, 1],
                                 padding='SAME',
                                 hidden_sizes=hidden_sizes,
                                 output_dim=output_dim,
                                 std_share_network=True)
        model.build(self._input_ph)
        with tf.compat.v1.variable_scope(model.name, reuse=True):
            std_share_output_weights = tf.compat.v1.get_variable(
                'dist_params/mean_std_network/output/kernel')
            std_share_output_bias = tf.compat.v1.get_variable(
                'dist_params/mean_std_network/output/bias')
        assert std_share_output_weights.shape[1] == output_dim * 2
        assert std_share_output_bias.shape == output_dim * 2

    @mock.patch('tensorflow.random.normal')
    @pytest.mark.parametrize('filters, in_channels, strides, output_dim, '
                             'hidden_sizes',
                             [(((3, (1, 1)), ), (3, ), (1, ), 1, (1, )),
                              (((3, (3, 3)), ), (3, ), (1, ), 2, (2, )),
                              (((3, (3, 3)), ), (3, ), (2, ), 1, (1, 1)),
                              (((3, (1, 1)), (3, (1, 1))), (3, 3), (1, 1), 2,
                               (2, 2)),
                              (((3, (3, 3)), (3, (3, 3))), (3, 3), (2, 2), 2,
                               (2, 2))])
    def test_without_std_share_network_output_values(self, mock_normal,
                                                     filters, in_channels,
                                                     strides, output_dim,
                                                     hidden_sizes):
        mock_normal.return_value = 0.5
        model = GaussianCNNModel(filters=filters,
                                 strides=strides,
                                 padding='VALID',
                                 output_dim=output_dim,
                                 init_std=2,
                                 hidden_sizes=hidden_sizes,
                                 std_share_network=False,
                                 adaptive_std=False,
                                 hidden_nonlinearity=None,
                                 std_parameterization='exp',
                                 hidden_w_init=tf.constant_initializer(0.01),
                                 output_w_init=tf.constant_initializer(1))
        outputs = model.build(self._input_ph).outputs

        action, mean, log_std, std_param = self.sess.run(
            outputs[:-1], feed_dict={self._input_ph: self.obs})

        filter_sum = 1

        for filter_iter, in_channel in zip(filters, in_channels):
            filter_height = filter_iter[1][0]
            filter_width = filter_iter[1][1]
            filter_sum *= 0.01 * filter_height * filter_width * in_channel

        for _ in hidden_sizes:
            filter_sum *= 0.01

        height_size = self.input_height
        width_size = self.input_width
        for filter_iter, stride in zip(filters, strides):
            height_size = int((height_size - filter_iter[1][0]) / stride) + 1
            width_size = int((width_size - filter_iter[1][1]) / stride) + 1
        flatten_shape = height_size * width_size * filters[-1][0]

        network_output = filter_sum * flatten_shape * np.prod(hidden_sizes)
        expected_mean = np.full((self.batch_size, output_dim),
                                network_output,
                                dtype=np.float32)
        expected_std_param = np.full((self.batch_size, output_dim),
                                     np.log(2),
                                     dtype=np.float32)
        expected_log_std = np.full((self.batch_size, output_dim),
                                   np.log(2),
                                   dtype=np.float32)

        assert np.allclose(mean, expected_mean)

        assert np.allclose(std_param, expected_std_param)
        assert np.allclose(log_std, expected_log_std)

        expected_action = 0.5 * np.exp(expected_log_std) + expected_mean
        assert np.allclose(action, expected_action, rtol=0, atol=0.1)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_without_std_share_network_shapes(self, output_dim, hidden_sizes):
        model = GaussianCNNModel(filters=((3, (3, 3)), (6, (3, 3))),
                                 strides=[1, 1],
                                 padding='SAME',
                                 hidden_sizes=hidden_sizes,
                                 output_dim=output_dim,
                                 std_share_network=False,
                                 adaptive_std=False)
        model.build(self._input_ph)
        with tf.compat.v1.variable_scope(model.name, reuse=True):
            mean_output_weights = tf.compat.v1.get_variable(
                'dist_params/mean_network/output/kernel')
            mean_output_bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/output/bias')
            log_std_output_weights = tf.compat.v1.get_variable(
                'dist_params/log_std_network/parameter')
        assert mean_output_weights.shape[1] == output_dim
        assert mean_output_bias.shape == output_dim
        assert log_std_output_weights.shape == output_dim

    @mock.patch('tensorflow.random.normal')
    @pytest.mark.parametrize('filters, in_channels, strides, output_dim, '
                             'hidden_sizes',
                             [(((3, (1, 1)), ), (3, ), (1, ), 1, (1, )),
                              (((3, (3, 3)), ), (3, ), (1, ), 2, (2, )),
                              (((3, (3, 3)), ), (3, ), (2, ), 1, (1, 1)),
                              (((3, (1, 1)), (3, (1, 1))), (3, 3), (1, 1), 2,
                               (2, 2)),
                              (((3, (3, 3)), (3, (3, 3))), (3, 3), (2, 2), 2,
                               (2, 2))])
    def test_adaptive_std_network_output_values(self, mock_normal, filters,
                                                in_channels, strides,
                                                output_dim, hidden_sizes):
        mock_normal.return_value = 0.5
        model = GaussianCNNModel(
            filters=filters,
            strides=strides,
            padding='VALID',
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            std_share_network=False,
            adaptive_std=True,
            hidden_nonlinearity=None,
            std_hidden_nonlinearity=None,
            std_filters=filters,
            std_strides=strides,
            std_padding='VALID',
            std_hidden_sizes=hidden_sizes,
            hidden_w_init=tf.constant_initializer(0.01),
            output_w_init=tf.constant_initializer(1),
            std_hidden_w_init=tf.constant_initializer(0.01),
            std_output_w_init=tf.constant_initializer(1))
        outputs = model.build(self._input_ph).outputs

        action, mean, log_std, std_param = self.sess.run(
            outputs[:-1], feed_dict={self._input_ph: self.obs})

        filter_sum = 1

        for filter_iter, in_channel in zip(filters, in_channels):
            filter_height = filter_iter[1][0]
            filter_width = filter_iter[1][1]
            filter_sum *= 0.01 * filter_height * filter_width * in_channel

        for _ in hidden_sizes:
            filter_sum *= 0.01

        height_size = self.input_height
        width_size = self.input_width
        for filter_iter, stride in zip(filters, strides):
            height_size = int((height_size - filter_iter[1][0]) / stride) + 1
            width_size = int((width_size - filter_iter[1][1]) / stride) + 1
        flatten_shape = height_size * width_size * filters[-1][0]

        network_output = filter_sum * flatten_shape * np.prod(hidden_sizes)
        expected_mean = np.full((self.batch_size, output_dim),
                                network_output,
                                dtype=np.float32)
        expected_std_param = np.full((self.batch_size, output_dim),
                                     network_output,
                                     dtype=np.float32)
        expected_log_std = np.full((self.batch_size, output_dim),
                                   network_output,
                                   dtype=np.float32)

        assert np.allclose(mean, expected_mean)

        assert np.allclose(std_param, expected_std_param)
        assert np.allclose(log_std, expected_log_std)

        expected_action = 0.5 * np.exp(expected_log_std) + expected_mean
        assert np.allclose(action, expected_action, rtol=0, atol=0.1)

    @pytest.mark.parametrize('output_dim, hidden_sizes, std_hidden_sizes',
                             [(1, (0, ), (0, )), (1, (1, ), (1, )),
                              (1, (2, ), (2, )), (2, (3, ), (3, )),
                              (2, (1, 1), (1, 1)), (3, (2, 2), (2, 2))])
    def test_adaptive_std_output_shape(self, output_dim, hidden_sizes,
                                       std_hidden_sizes):
        model = GaussianCNNModel(
            filters=((3, (3, 3)), (6, (3, 3))),
            strides=[1, 1],
            padding='SAME',
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            std_share_network=False,
            adaptive_std=True,
            hidden_nonlinearity=None,
            std_hidden_nonlinearity=None,
            std_filters=((3, (3, 3)), (6, (3, 3))),
            std_strides=[1, 1],
            std_padding='SAME',
            std_hidden_sizes=std_hidden_sizes,
            hidden_w_init=tf.constant_initializer(0.01),
            output_w_init=tf.constant_initializer(1),
            std_hidden_w_init=tf.constant_initializer(0.01),
            std_output_w_init=tf.constant_initializer(1))

        model.build(self._input_ph)
        with tf.compat.v1.variable_scope(model.name, reuse=True):
            mean_output_weights = tf.compat.v1.get_variable(
                'dist_params/mean_network/output/kernel')
            mean_output_bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/output/bias')
            log_std_output_weights = tf.compat.v1.get_variable(
                'dist_params/log_std_network/output/kernel')
            log_std_output_bias = tf.compat.v1.get_variable(
                'dist_params/log_std_network/output/bias')

        assert mean_output_weights.shape[1] == output_dim
        assert mean_output_bias.shape == output_dim
        assert log_std_output_weights.shape[1] == output_dim
        assert log_std_output_bias.shape == output_dim

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    @mock.patch('tensorflow.random.normal')
    def test_std_share_network_is_pickleable(self, mock_normal, output_dim,
                                             hidden_sizes):
        mock_normal.return_value = 0.5
        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, 10, 10, 3))
        model = GaussianCNNModel(filters=((3, (3, 3)), (6, (3, 3))),
                                 strides=[1, 1],
                                 padding='SAME',
                                 hidden_sizes=hidden_sizes,
                                 output_dim=output_dim,
                                 std_share_network=True)
        outputs = model.build(input_var).outputs

        # get output bias
        with tf.compat.v1.variable_scope('GaussianCNNModel', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_std_network/output/bias')
        # assign it to all ones
        bias.load(tf.ones_like(bias).eval())

        output1 = self.sess.run(outputs[:-1], feed_dict={input_var: self.obs})

        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, 10, 10, 3))
            model_pickled = pickle.loads(h)
            outputs = model_pickled.build(input_var).outputs
            output2 = sess.run(outputs[:-1], feed_dict={input_var: self.obs})

            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    @mock.patch('tensorflow.random.normal')
    def test_without_std_share_network_is_pickleable(self, mock_normal,
                                                     output_dim, hidden_sizes):
        mock_normal.return_value = 0.5
        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, 10, 10, 3))
        model = GaussianCNNModel(filters=((3, (3, 3)), (6, (3, 3))),
                                 strides=[1, 1],
                                 padding='SAME',
                                 hidden_sizes=hidden_sizes,
                                 output_dim=output_dim,
                                 std_share_network=False,
                                 adaptive_std=False)
        outputs = model.build(input_var).outputs

        # get output bias
        with tf.compat.v1.variable_scope('GaussianCNNModel', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/output/bias')
        # assign it to all ones
        bias.load(tf.ones_like(bias).eval())

        output1 = self.sess.run(outputs[:-1], feed_dict={input_var: self.obs})

        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, 10, 10, 3))
            model_pickled = pickle.loads(h)
            outputs = model_pickled.build(input_var).outputs
            output2 = sess.run(outputs[:-1], feed_dict={input_var: self.obs})
            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('output_dim, hidden_sizes, std_hidden_sizes',
                             [(1, (0, ), (0, )), (1, (1, ), (1, )),
                              (1, (2, ), (2, )), (2, (3, ), (3, )),
                              (2, (1, 1), (1, 1)), (3, (2, 2), (2, 2))])
    @mock.patch('tensorflow.random.normal')
    def test_adaptive_std_is_pickleable(self, mock_normal, output_dim,
                                        hidden_sizes, std_hidden_sizes):
        mock_normal.return_value = 0.5
        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, 10, 10, 3))
        model = GaussianCNNModel(filters=((3, (3, 3)), (6, (3, 3))),
                                 strides=[1, 1],
                                 padding='SAME',
                                 output_dim=output_dim,
                                 hidden_sizes=hidden_sizes,
                                 std_share_network=False,
                                 adaptive_std=True,
                                 hidden_nonlinearity=None,
                                 std_hidden_nonlinearity=None,
                                 std_filters=((3, (3, 3)), (6, (3, 3))),
                                 std_strides=[1, 1],
                                 std_padding='SAME',
                                 std_hidden_sizes=std_hidden_sizes,
                                 hidden_w_init=tf.constant_initializer(1),
                                 output_w_init=tf.constant_initializer(1),
                                 std_hidden_w_init=tf.constant_initializer(1),
                                 std_output_w_init=tf.constant_initializer(1))
        outputs = model.build(input_var).outputs

        # get output bias
        with tf.compat.v1.variable_scope('GaussianCNNModel', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/output/bias')
        # assign it to all ones
        bias.load(tf.ones_like(bias).eval())

        h = pickle.dumps(model)
        output1 = self.sess.run(outputs[:-1], feed_dict={input_var: self.obs})
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, 10, 10, 3))
            model_pickled = pickle.loads(h)
            outputs = model_pickled.build(input_var).outputs
            output2 = sess.run(outputs[:-1], feed_dict={input_var: self.obs})
            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    @mock.patch('tensorflow.random.normal')
    def test_softplus_output_values(self, mock_normal, output_dim,
                                    hidden_sizes):
        mock_normal.return_value = 0.5
        filters = ((3, (3, 3)), (6, (3, 3)))
        in_channels = [3, 3]
        strides = [1, 1]

        model = GaussianCNNModel(filters=filters,
                                 strides=strides,
                                 padding='VALID',
                                 output_dim=output_dim,
                                 init_std=2.0,
                                 hidden_sizes=hidden_sizes,
                                 std_share_network=False,
                                 adaptive_std=False,
                                 hidden_nonlinearity=None,
                                 std_parameterization='softplus',
                                 hidden_w_init=tf.constant_initializer(0.01),
                                 output_w_init=tf.constant_initializer(1))
        outputs = model.build(self._input_ph).outputs

        action, mean, log_std, std_param = self.sess.run(
            outputs[:-1], feed_dict={self._input_ph: self.obs})

        filter_sum = 1
        for filter_iter, in_channel in zip(filters, in_channels):
            filter_height = filter_iter[1][0]
            filter_width = filter_iter[1][1]
            filter_sum *= 0.01 * filter_height * filter_width * in_channel

        for _ in hidden_sizes:
            filter_sum *= 0.01

        height_size = self.input_height
        width_size = self.input_width
        for filter_iter, stride in zip(filters, strides):
            height_size = int((height_size - filter_iter[1][0]) / stride) + 1
            width_size = int((width_size - filter_iter[1][1]) / stride) + 1
        flatten_shape = height_size * width_size * filters[-1][0]

        network_output = filter_sum * flatten_shape * np.prod(hidden_sizes)
        expected_mean = np.full((self.batch_size, output_dim),
                                network_output,
                                dtype=np.float32)
        expected_std_param = np.full((self.batch_size, output_dim),
                                     np.log(np.exp(2) - 1),
                                     dtype=np.float32)
        expected_log_std = np.log(np.log(1. + np.exp(expected_std_param)))

        assert np.allclose(mean, expected_mean)
        assert np.allclose(std_param, expected_std_param)
        assert np.allclose(log_std, expected_log_std)

        expected_action = 0.5 * np.exp(expected_log_std) + expected_mean
        assert np.allclose(action, expected_action, rtol=0, atol=0.1)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_exp_min_std(self, output_dim, hidden_sizes):
        filters = ((3, (3, 3)), (6, (3, 3)))
        strides = [1, 1]

        model = GaussianCNNModel(filters=filters,
                                 strides=strides,
                                 padding='VALID',
                                 output_dim=output_dim,
                                 init_std=2.0,
                                 hidden_sizes=hidden_sizes,
                                 std_share_network=False,
                                 adaptive_std=False,
                                 hidden_nonlinearity=None,
                                 std_parameterization='exp',
                                 min_std=10,
                                 hidden_w_init=tf.constant_initializer(0.01),
                                 output_w_init=tf.constant_initializer(1))
        outputs = model.build(self._input_ph).outputs
        _, _, log_std, std_param = self.sess.run(
            outputs[:-1], feed_dict={self._input_ph: self.obs})

        expected_log_std = np.full([1, output_dim], np.log(10))
        expected_std_param = np.full([1, output_dim], np.log(10))
        assert np.allclose(log_std, expected_log_std)
        assert np.allclose(std_param, expected_std_param)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_exp_max_std(self, output_dim, hidden_sizes):
        filters = ((3, (3, 3)), (6, (3, 3)))
        strides = [1, 1]

        model = GaussianCNNModel(filters=filters,
                                 strides=strides,
                                 padding='VALID',
                                 output_dim=output_dim,
                                 init_std=10.0,
                                 hidden_sizes=hidden_sizes,
                                 std_share_network=False,
                                 adaptive_std=False,
                                 hidden_nonlinearity=None,
                                 std_parameterization='exp',
                                 max_std=1.0,
                                 hidden_w_init=tf.constant_initializer(0.01),
                                 output_w_init=tf.constant_initializer(1))
        outputs = model.build(self._input_ph).outputs
        _, _, log_std, std_param = self.sess.run(
            outputs[:-1], feed_dict={self._input_ph: self.obs})

        expected_log_std = np.full([1, output_dim], np.log(1))
        expected_std_param = np.full([1, output_dim], np.log(1))
        assert np.allclose(log_std, expected_log_std)
        assert np.allclose(std_param, expected_std_param)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_softplus_min_std(self, output_dim, hidden_sizes):
        filters = ((3, (3, 3)), (6, (3, 3)))
        strides = [1, 1]

        model = GaussianCNNModel(filters=filters,
                                 strides=strides,
                                 padding='VALID',
                                 output_dim=output_dim,
                                 init_std=2.0,
                                 hidden_sizes=hidden_sizes,
                                 std_share_network=False,
                                 adaptive_std=False,
                                 hidden_nonlinearity=None,
                                 std_parameterization='softplus',
                                 min_std=10,
                                 hidden_w_init=tf.constant_initializer(0.1),
                                 output_w_init=tf.constant_initializer(1))
        outputs = model.build(self._input_ph).outputs
        _, _, log_std, std_param = self.sess.run(
            outputs[:-1], feed_dict={self._input_ph: self.obs})

        expected_log_std = np.full([1, output_dim], np.log(10))
        expected_std_param = np.full([1, output_dim], np.log(np.exp(10) - 1))

        assert np.allclose(log_std, expected_log_std)
        assert np.allclose(std_param, expected_std_param)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_softplus_max_std(self, output_dim, hidden_sizes):
        filters = ((3, (3, 3)), (6, (3, 3)))
        strides = [1, 1]

        model = GaussianCNNModel(filters=filters,
                                 strides=strides,
                                 padding='VALID',
                                 output_dim=output_dim,
                                 init_std=10.0,
                                 hidden_sizes=hidden_sizes,
                                 std_share_network=False,
                                 adaptive_std=False,
                                 hidden_nonlinearity=None,
                                 std_parameterization='softplus',
                                 max_std=1.0,
                                 hidden_w_init=tf.constant_initializer(0.1),
                                 output_w_init=tf.constant_initializer(1))
        outputs = model.build(self._input_ph).outputs
        _, _, log_std, std_param = self.sess.run(
            outputs[:-1], feed_dict={self._input_ph: self.obs})

        expected_log_std = np.full([1, output_dim], np.log(1))
        expected_std_param = np.full([1, output_dim], np.log(np.exp(1) - 1))

        assert np.allclose(log_std, expected_log_std, rtol=0, atol=0.0001)
        assert np.allclose(std_param, expected_std_param, rtol=0, atol=0.0001)

    def test_unknown_std_parameterization(self):
        with pytest.raises(NotImplementedError):
            _ = GaussianCNNModel(filters=(((3, 3), 3), ((3, 3), 6)),
                                 strides=[1, 1],
                                 padding='SAME',
                                 hidden_sizes=(1, ),
                                 output_dim=1,
                                 std_parameterization='unknown')
