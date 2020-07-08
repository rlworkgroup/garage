import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from garage.tf.models import GaussianLSTMModel

from tests.fixtures import TfGraphTestCase
from tests.helpers import recurrent_step_lstm


class TestGaussianLSTMModel(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.batch_size = 1
        self.time_step = 2
        self.feature_shape = 2
        self.default_initializer = tf.constant_initializer(0.1)

        self.obs_inputs = np.full(
            (self.batch_size, self.time_step, self.feature_shape), 1.)
        self.obs_input = np.full((self.batch_size, self.feature_shape), 1.)

        self.input_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, None,
                                                         self.feature_shape),
                                                  name='input')
        self.step_input_var = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.feature_shape), name='step_input')

    def test_dist(self):
        model = GaussianLSTMModel(output_dim=1, hidden_dim=1)
        step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size, 1),
                                                   name='step_hidden',
                                                   dtype=tf.float32)
        step_cell_var = tf.compat.v1.placeholder(shape=(self.batch_size, 1),
                                                 name='step_cell',
                                                 dtype=tf.float32)
        dist = model.build(self.input_var, self.step_input_var,
                           step_hidden_var, step_cell_var).dist
        assert isinstance(dist, tfp.distributions.MultivariateNormalDiag)

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_dim', [
        (1, 1),
        (2, 2),
        (3, 3)
    ])
    # yapf: enable
    @mock.patch('tensorflow.random.normal')
    def test_std_share_network_output_values(self, mock_normal, output_dim,
                                             hidden_dim):
        mock_normal.return_value = 0.5
        model = GaussianLSTMModel(output_dim=output_dim,
                                  hidden_dim=hidden_dim,
                                  std_share_network=True,
                                  hidden_nonlinearity=None,
                                  recurrent_nonlinearity=None,
                                  hidden_w_init=self.default_initializer,
                                  recurrent_w_init=self.default_initializer,
                                  output_w_init=self.default_initializer)
        step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                          hidden_dim),
                                                   name='step_hidden',
                                                   dtype=tf.float32)
        step_cell_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                        hidden_dim),
                                                 name='step_cell',
                                                 dtype=tf.float32)
        (_, step_mean_var, step_log_std_var, step_hidden, step_cell,
         hidden_init_var, cell_init_var) = model.build(self.input_var,
                                                       self.step_input_var,
                                                       step_hidden_var,
                                                       step_cell_var).outputs

        hidden1 = hidden2 = np.full((self.batch_size, hidden_dim),
                                    hidden_init_var.eval())
        cell1 = cell2 = np.full((self.batch_size, hidden_dim),
                                cell_init_var.eval())

        for _ in range(self.time_step):
            mean1, log_std1, hidden1, cell1 = self.sess.run(
                [step_mean_var, step_log_std_var, step_hidden, step_cell],
                feed_dict={
                    self.step_input_var: self.obs_input,
                    step_hidden_var: hidden1,
                    step_cell_var: cell1
                })

            hidden2, cell2 = recurrent_step_lstm(input_val=self.obs_input,
                                                 num_units=hidden_dim,
                                                 step_hidden=hidden2,
                                                 step_cell=cell2,
                                                 w_x_init=0.1,
                                                 w_h_init=0.1,
                                                 b_init=0.,
                                                 nonlinearity=None,
                                                 gate_nonlinearity=None)

            output_nonlinearity = np.full(
                (np.prod(hidden2.shape[1:]), output_dim), 0.1)
            output2 = np.matmul(hidden2, output_nonlinearity)
            assert np.allclose(mean1, output2)
            assert np.allclose(log_std1, output2)
            assert np.allclose(hidden1, hidden2)
            assert np.allclose(cell1, cell2)

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_dim', [
        (1, 1),
        (2, 2),
        (3, 3)
    ])
    # yapf: enable
    def test_std_share_network_shapes(self, output_dim, hidden_dim):
        model = GaussianLSTMModel(output_dim=output_dim,
                                  hidden_dim=hidden_dim,
                                  std_share_network=True,
                                  hidden_nonlinearity=None,
                                  recurrent_nonlinearity=None,
                                  hidden_w_init=self.default_initializer,
                                  recurrent_w_init=self.default_initializer,
                                  output_w_init=self.default_initializer)
        step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                          hidden_dim),
                                                   name='step_hidden',
                                                   dtype=tf.float32)
        step_cell_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                        hidden_dim),
                                                 name='step_cell',
                                                 dtype=tf.float32)
        model.build(self.input_var, self.step_input_var, step_hidden_var,
                    step_cell_var)

        # output layer is a tf.keras.layers.Dense object,
        # which cannot be access by tf.compat.v1.variable_scope.
        # A workaround is to access in tf.compat.v1.global_variables()
        for var in tf.compat.v1.global_variables():
            if 'output_layer/kernel' in var.name:
                std_share_output_weights = var
            if 'output_layer/bias' in var.name:
                std_share_output_bias = var
        assert std_share_output_weights.shape[1] == output_dim * 2
        assert std_share_output_bias.shape == output_dim * 2

    @pytest.mark.parametrize('output_dim, hidden_dim, init_std', [(1, 1, 1),
                                                                  (1, 1, 2),
                                                                  (1, 2, 1),
                                                                  (1, 2, 2),
                                                                  (3, 3, 1),
                                                                  (3, 3, 2)])
    @mock.patch('tensorflow.random.normal')
    def test_without_std_share_network_output_values(self, mock_normal,
                                                     output_dim, hidden_dim,
                                                     init_std):
        mock_normal.return_value = 0.5
        model = GaussianLSTMModel(output_dim=output_dim,
                                  hidden_dim=hidden_dim,
                                  std_share_network=False,
                                  hidden_nonlinearity=None,
                                  recurrent_nonlinearity=None,
                                  hidden_w_init=self.default_initializer,
                                  recurrent_w_init=self.default_initializer,
                                  output_w_init=self.default_initializer,
                                  init_std=init_std)
        step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                          hidden_dim),
                                                   name='step_hidden',
                                                   dtype=tf.float32)
        step_cell_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                        hidden_dim),
                                                 name='step_cell',
                                                 dtype=tf.float32)
        (_, step_mean_var, step_log_std_var, step_hidden, step_cell,
         hidden_init_var, cell_init_var) = model.build(self.input_var,
                                                       self.step_input_var,
                                                       step_hidden_var,
                                                       step_cell_var).outputs

        hidden1 = hidden2 = np.full((self.batch_size, hidden_dim),
                                    hidden_init_var.eval())
        cell1 = cell2 = np.full((self.batch_size, hidden_dim),
                                cell_init_var.eval())

        for _ in range(self.time_step):
            mean1, log_std1, hidden1, cell1 = self.sess.run(
                [step_mean_var, step_log_std_var, step_hidden, step_cell],
                feed_dict={
                    self.step_input_var: self.obs_input,
                    step_hidden_var: hidden1,
                    step_cell_var: cell1
                })

            hidden2, cell2 = recurrent_step_lstm(input_val=self.obs_input,
                                                 num_units=hidden_dim,
                                                 step_hidden=hidden2,
                                                 step_cell=cell2,
                                                 w_x_init=0.1,
                                                 w_h_init=0.1,
                                                 b_init=0.,
                                                 nonlinearity=None,
                                                 gate_nonlinearity=None)

            output_nonlinearity = np.full(
                (np.prod(hidden2.shape[1:]), output_dim), 0.1)
            output2 = np.matmul(hidden2, output_nonlinearity)
            assert np.allclose(mean1, output2)
            expected_log_std = np.full((self.batch_size, output_dim),
                                       np.log(init_std))
            assert np.allclose(log_std1, expected_log_std)
            assert np.allclose(hidden1, hidden2)
            assert np.allclose(cell1, cell2)

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_dim', [
        (1, 1),
        (2, 2),
        (3, 3)
    ])
    # yapf: enable
    def test_without_std_share_network_shapes(self, output_dim, hidden_dim):
        model = GaussianLSTMModel(output_dim=output_dim,
                                  hidden_dim=hidden_dim,
                                  std_share_network=False,
                                  hidden_nonlinearity=None,
                                  recurrent_nonlinearity=None,
                                  hidden_w_init=self.default_initializer,
                                  recurrent_w_init=self.default_initializer,
                                  output_w_init=self.default_initializer)
        step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                          hidden_dim),
                                                   name='step_hidden',
                                                   dtype=tf.float32)
        step_cell_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                        hidden_dim),
                                                 name='step_cell',
                                                 dtype=tf.float32)
        model.build(self.input_var, self.step_input_var, step_hidden_var,
                    step_cell_var)

        # output layer is a tf.keras.layers.Dense object,
        # which cannot be access by tf.compat.v1.variable_scope.
        # A workaround is to access in tf.compat.v1.global_variables()
        for var in tf.compat.v1.global_variables():
            if 'output_layer/kernel' in var.name:
                std_share_output_weights = var
            if 'output_layer/bias' in var.name:
                std_share_output_bias = var
            if 'log_std_param/parameter' in var.name:
                log_std_param = var
        assert std_share_output_weights.shape[1] == output_dim
        assert std_share_output_bias.shape == output_dim
        assert log_std_param.shape == output_dim

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_dim', [
        (1, 1),
        (2, 2),
        (3, 3)
    ])
    # yapf: enable
    @mock.patch('tensorflow.random.normal')
    def test_std_share_network_is_pickleable(self, mock_normal, output_dim,
                                             hidden_dim):
        mock_normal.return_value = 0.5
        model = GaussianLSTMModel(output_dim=output_dim,
                                  hidden_dim=hidden_dim,
                                  std_share_network=True,
                                  hidden_nonlinearity=None,
                                  recurrent_nonlinearity=None,
                                  hidden_w_init=self.default_initializer,
                                  recurrent_w_init=self.default_initializer,
                                  output_w_init=self.default_initializer)
        step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                          hidden_dim),
                                                   name='step_hidden',
                                                   dtype=tf.float32)
        step_cell_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                        hidden_dim),
                                                 name='step_cell',
                                                 dtype=tf.float32)
        (dist, step_mean_var, step_log_std_var, step_hidden, step_cell, _,
         _) = model.build(self.input_var, self.step_input_var, step_hidden_var,
                          step_cell_var).outputs

        # output layer is a tf.keras.layers.Dense object,
        # which cannot be access by tf.compat.v1.variable_scope.
        # A workaround is to access in tf.compat.v1.global_variables()
        for var in tf.compat.v1.global_variables():
            if 'output_layer/bias' in var.name:
                var.load(tf.ones_like(var).eval())

        hidden = np.zeros((self.batch_size, hidden_dim))
        cell = np.zeros((self.batch_size, hidden_dim))

        outputs1 = self.sess.run([dist.loc, dist.scale.diag],
                                 feed_dict={self.input_var: self.obs_inputs})
        output1 = self.sess.run(
            [step_mean_var, step_log_std_var, step_hidden, step_cell],
            feed_dict={
                self.step_input_var: self.obs_input,
                step_hidden_var: hidden,
                step_cell_var: cell
            })

        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            model_pickled = pickle.loads(h)

            input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, None,
                                                        self.feature_shape),
                                                 name='input')
            step_input_var = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, self.feature_shape),
                name='step_input')
            step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                              hidden_dim),
                                                       name='initial_hidden',
                                                       dtype=tf.float32)
            step_cell_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                            hidden_dim),
                                                     name='initial_cell',
                                                     dtype=tf.float32)

            (dist2, step_mean_var2, step_log_std_var2, step_hidden2,
             step_cell2, _, _) = model_pickled.build(input_var, step_input_var,
                                                     step_hidden_var,
                                                     step_cell_var).outputs

            outputs2 = sess.run([dist2.loc, dist2.scale.diag],
                                feed_dict={input_var: self.obs_inputs})
            output2 = sess.run(
                [step_mean_var2, step_log_std_var2, step_hidden2, step_cell2],
                feed_dict={
                    step_input_var: self.obs_input,
                    step_hidden_var: hidden,
                    step_cell_var: cell
                })
            assert np.array_equal(outputs1, outputs2)
            assert np.array_equal(output1, output2)

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_dim', [
        (1, 1),
        (2, 2),
        (3, 3)
    ])
    # yapf: enable
    @mock.patch('tensorflow.random.normal')
    def test_without_std_share_network_is_pickleable(self, mock_normal,
                                                     output_dim, hidden_dim):
        mock_normal.return_value = 0.5
        model = GaussianLSTMModel(output_dim=output_dim,
                                  hidden_dim=hidden_dim,
                                  std_share_network=False,
                                  hidden_nonlinearity=None,
                                  recurrent_nonlinearity=None,
                                  hidden_w_init=self.default_initializer,
                                  recurrent_w_init=self.default_initializer,
                                  output_w_init=self.default_initializer)
        step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                          hidden_dim),
                                                   name='step_hidden',
                                                   dtype=tf.float32)
        step_cell_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                        hidden_dim),
                                                 name='step_cell',
                                                 dtype=tf.float32)
        (dist, step_mean_var, step_log_std_var, step_hidden, step_cell, _,
         _) = model.build(self.input_var, self.step_input_var, step_hidden_var,
                          step_cell_var).outputs

        # output layer is a tf.keras.layers.Dense object,
        # which cannot be access by tf.compat.v1.variable_scope.
        # A workaround is to access in tf.compat.v1.global_variables()
        for var in tf.compat.v1.global_variables():
            if 'output_layer/bias' in var.name:
                var.load(tf.ones_like(var).eval())

        hidden = np.zeros((self.batch_size, hidden_dim))
        cell = np.zeros((self.batch_size, hidden_dim))

        outputs1 = self.sess.run([dist.loc, dist.scale.diag],
                                 feed_dict={self.input_var: self.obs_inputs})
        output1 = self.sess.run(
            [step_mean_var, step_log_std_var, step_hidden, step_cell],
            feed_dict={
                self.step_input_var: self.obs_input,
                step_hidden_var: hidden,
                step_cell_var: cell
            })

        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            model_pickled = pickle.loads(h)

            input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, None,
                                                        self.feature_shape),
                                                 name='input')
            step_input_var = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, self.feature_shape),
                name='step_input')
            step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                              hidden_dim),
                                                       name='initial_hidden',
                                                       dtype=tf.float32)
            step_cell_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                            hidden_dim),
                                                     name='initial_cell',
                                                     dtype=tf.float32)

            (dist2, step_mean_var2, step_log_std_var2, step_hidden2,
             step_cell2, _, _) = model_pickled.build(input_var, step_input_var,
                                                     step_hidden_var,
                                                     step_cell_var).outputs

            outputs2 = sess.run([dist2.loc, dist2.scale.diag],
                                feed_dict={input_var: self.obs_inputs})
            output2 = sess.run(
                [step_mean_var2, step_log_std_var2, step_hidden2, step_cell2],
                feed_dict={
                    step_input_var: self.obs_input,
                    step_hidden_var: hidden,
                    step_cell_var: cell
                })
            assert np.array_equal(outputs1, outputs2)
            assert np.array_equal(output1, output2)
