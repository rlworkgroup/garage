import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.models import GaussianGRUModel
from tests.fixtures import TfGraphTestCase
from tests.helpers import recurrent_step_gru


class TestGaussianGRUModel(TfGraphTestCase):

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
        model = GaussianGRUModel(output_dim=output_dim,
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
        (mean_var, step_mean_var, log_std_var, step_log_std_var, step_hidden,
         hidden_init_var, dist) = model.build(self.input_var,
                                              self.step_input_var,
                                              step_hidden_var)

        hidden1 = hidden2 = np.full((self.batch_size, hidden_dim),
                                    hidden_init_var.eval())

        mean, log_std = self.sess.run(
            [mean_var, log_std_var],
            feed_dict={self.input_var: self.obs_inputs})

        for i in range(self.time_step):
            mean1, log_std1, hidden1 = self.sess.run(
                [step_mean_var, step_log_std_var, step_hidden],
                feed_dict={
                    self.step_input_var: self.obs_input,
                    step_hidden_var: hidden1
                })

            hidden2 = recurrent_step_gru(input_val=self.obs_input,
                                         num_units=hidden_dim,
                                         step_hidden=hidden2,
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

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_dim', [
        (1, 1),
        (2, 2),
        (3, 3)
    ])
    # yapf: enable
    def test_std_share_network_shapes(self, output_dim, hidden_dim):
        model = GaussianGRUModel(output_dim=output_dim,
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
        (mean_var, step_mean_var, log_std_var, step_log_std_var, step_hidden,
         hidden_init_var, dist) = model.build(self.input_var,
                                              self.step_input_var,
                                              step_hidden_var)

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

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_dim, init_std', [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 1),
        (1, 2, 2),
        (3, 3, 1),
        (3, 3, 2),
    ])
    # yapf: enable
    @mock.patch('tensorflow.random.normal')
    def test_without_std_share_network_output_values(self, mock_normal,
                                                     output_dim, hidden_dim,
                                                     init_std):
        mock_normal.return_value = 0.5
        model = GaussianGRUModel(output_dim=output_dim,
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
        (mean_var, step_mean_var, log_std_var, step_log_std_var, step_hidden,
         hidden_init_var, dist) = model.build(self.input_var,
                                              self.step_input_var,
                                              step_hidden_var)

        hidden1 = hidden2 = np.full((self.batch_size, hidden_dim),
                                    hidden_init_var.eval())

        mean, log_std = self.sess.run(
            [mean_var, log_std_var],
            feed_dict={self.input_var: self.obs_inputs})

        for i in range(self.time_step):
            mean1, log_std1, hidden1 = self.sess.run(
                [step_mean_var, step_log_std_var, step_hidden],
                feed_dict={
                    self.step_input_var: self.obs_input,
                    step_hidden_var: hidden1
                })

            hidden2 = recurrent_step_gru(input_val=self.obs_input,
                                         num_units=hidden_dim,
                                         step_hidden=hidden2,
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

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_dim', [
        (1, 1),
        (2, 2),
        (3, 3)
    ])
    # yapf: enable
    def test_without_std_share_network_shapes(self, output_dim, hidden_dim):
        model = GaussianGRUModel(output_dim=output_dim,
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
        (mean_var, step_mean_var, log_std_var, step_log_std_var, step_hidden,
         hidden_init_var, dist) = model.build(self.input_var,
                                              self.step_input_var,
                                              step_hidden_var)

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
        model = GaussianGRUModel(output_dim=output_dim,
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
        (mean_var, step_mean_var, log_std_var, step_log_std_var, step_hidden,
         _, _) = model.build(self.input_var, self.step_input_var,
                             step_hidden_var)

        # output layer is a tf.keras.layers.Dense object,
        # which cannot be access by tf.compat.v1.variable_scope.
        # A workaround is to access in tf.compat.v1.global_variables()
        for var in tf.compat.v1.global_variables():
            if 'output_layer/bias' in var.name:
                var.load(tf.ones_like(var).eval())

        hidden = np.zeros((self.batch_size, hidden_dim))

        outputs1 = self.sess.run([mean_var, log_std_var],
                                 feed_dict={self.input_var: self.obs_inputs})
        output1 = self.sess.run([step_mean_var, step_log_std_var, step_hidden],
                                feed_dict={
                                    self.step_input_var: self.obs_input,
                                    step_hidden_var: hidden
                                })  # noqa: E126

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

            (mean_var2, step_mean_var2, log_std_var2, step_log_std_var2,
             step_hidden2, _, _) = model_pickled.build(input_var,
                                                       step_input_var,
                                                       step_hidden_var)

            outputs2 = sess.run([mean_var2, log_std_var2],
                                feed_dict={input_var: self.obs_inputs})
            output2 = sess.run(
                [step_mean_var2, step_log_std_var2, step_hidden2],
                feed_dict={
                    step_input_var: self.obs_input,
                    step_hidden_var: hidden
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
        model = GaussianGRUModel(output_dim=output_dim,
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
        (mean_var, step_mean_var, log_std_var, step_log_std_var, step_hidden,
         _, _) = model.build(self.input_var, self.step_input_var,
                             step_hidden_var)

        # output layer is a tf.keras.layers.Dense object,
        # which cannot be access by tf.compat.v1.variable_scope.
        # A workaround is to access in tf.compat.v1.global_variables()
        for var in tf.compat.v1.global_variables():
            if 'output_layer/bias' in var.name:
                var.load(tf.ones_like(var).eval())

        hidden = np.zeros((self.batch_size, hidden_dim))

        outputs1 = self.sess.run([mean_var, log_std_var],
                                 feed_dict={self.input_var: self.obs_inputs})
        output1 = self.sess.run([step_mean_var, step_log_std_var, step_hidden],
                                feed_dict={
                                    self.step_input_var: self.obs_input,
                                    step_hidden_var: hidden
                                })  # noqa: E126

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

            (mean_var2, step_mean_var2, log_std_var2, step_log_std_var2,
             step_hidden2, _, _) = model_pickled.build(input_var,
                                                       step_input_var,
                                                       step_hidden_var)

            outputs2 = sess.run([mean_var2, log_std_var2],
                                feed_dict={input_var: self.obs_inputs})
            output2 = sess.run(
                [step_mean_var2, step_log_std_var2, step_hidden2],
                feed_dict={
                    step_input_var: self.obs_input,
                    step_hidden_var: hidden
                })
            assert np.array_equal(outputs1, outputs2)
            assert np.array_equal(output1, output2)
