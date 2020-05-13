import pickle

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from garage.tf.models import GaussianMLPModel2
from tests.fixtures import TfGraphTestCase


class TestGaussianMLPModel2(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.input_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, None, 5))
        self.obs = np.ones((1, 1, 5))

    def test_dist(self):
        model = GaussianMLPModel2(output_dim=1)
        dist = model.build(self.input_var)
        assert isinstance(dist, tfp.distributions.MultivariateNormalDiag)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_std_share_network_output_values(self, output_dim, hidden_sizes):
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  std_share_network=True,
                                  hidden_nonlinearity=None,
                                  std_parameterization='exp',
                                  hidden_w_init=tf.ones_initializer(),
                                  output_w_init=tf.ones_initializer())
        dist = model.build(self.input_var)

        mean, log_std = self.sess.run(
            [dist.loc, tf.math.log(dist.stddev())],
            feed_dict={self.input_var: self.obs})

        expected_mean = np.full([1, 1, output_dim], 5 * np.prod(hidden_sizes))
        expected_log_std = np.full([1, 1, output_dim],
                                   5 * np.prod(hidden_sizes))
        assert np.array_equal(mean, expected_mean)
        assert np.array_equal(log_std, expected_log_std)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_std_share_network_shapes(self, output_dim, hidden_sizes):
        # should be 2 * output_dim
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  std_share_network=True)
        model.build(self.input_var)
        with tf.compat.v1.variable_scope(model.name, reuse=True):
            std_share_output_weights = tf.compat.v1.get_variable(
                'dist_params/mean_std_network/output/kernel')
            std_share_output_bias = tf.compat.v1.get_variable(
                'dist_params/mean_std_network/output/bias')
        assert std_share_output_weights.shape[1] == output_dim * 2
        assert std_share_output_bias.shape == output_dim * 2

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_without_std_share_network_output_values(self, output_dim,
                                                     hidden_sizes):
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  init_std=2,
                                  std_share_network=False,
                                  adaptive_std=False,
                                  hidden_nonlinearity=None,
                                  hidden_w_init=tf.ones_initializer(),
                                  output_w_init=tf.ones_initializer())
        dist = model.build(self.input_var)

        mean, log_std = self.sess.run(
            [dist.loc, tf.math.log(dist.stddev())],
            feed_dict={self.input_var: self.obs})

        expected_mean = np.full([1, 1, output_dim], 5 * np.prod(hidden_sizes))
        expected_log_std = np.full([1, 1, output_dim], np.log(2.))
        assert np.array_equal(mean, expected_mean)
        assert np.allclose(log_std, expected_log_std)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_without_std_share_network_shapes(self, output_dim, hidden_sizes):
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  std_share_network=False,
                                  adaptive_std=False)
        model.build(self.input_var)
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

    @pytest.mark.parametrize('output_dim, hidden_sizes, std_hidden_sizes',
                             [(1, (0, ), (0, )), (1, (1, ), (1, )),
                              (1, (2, ), (2, )), (2, (3, ), (3, )),
                              (2, (1, 1), (1, 1)), (3, (2, 2), (2, 2))])
    def test_adaptive_std_network_output_values(self, output_dim, hidden_sizes,
                                                std_hidden_sizes):
        model = GaussianMLPModel2(output_dim=output_dim,
                                  std_share_network=False,
                                  hidden_sizes=hidden_sizes,
                                  std_hidden_sizes=std_hidden_sizes,
                                  adaptive_std=True,
                                  hidden_nonlinearity=None,
                                  hidden_w_init=tf.ones_initializer(),
                                  output_w_init=tf.ones_initializer(),
                                  std_hidden_nonlinearity=None,
                                  std_hidden_w_init=tf.ones_initializer(),
                                  std_output_w_init=tf.ones_initializer())
        dist = model.build(self.input_var)

        mean, log_std = self.sess.run(
            [dist.loc, tf.math.log(dist.stddev())],
            feed_dict={self.input_var: self.obs})

        expected_mean = np.full([1, 1, output_dim], 5 * np.prod(hidden_sizes))
        expected_log_std = np.full([1, 1, output_dim],
                                   5 * np.prod(std_hidden_sizes))
        assert np.array_equal(mean, expected_mean)
        assert np.array_equal(log_std, expected_log_std)

    @pytest.mark.parametrize('output_dim, hidden_sizes, std_hidden_sizes',
                             [(1, (0, ), (0, )), (1, (1, ), (1, )),
                              (1, (2, ), (2, )), (2, (3, ), (3, )),
                              (2, (1, 1), (1, 1)), (3, (2, 2), (2, 2))])
    def test_adaptive_std_output_shape(self, output_dim, hidden_sizes,
                                       std_hidden_sizes):
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  std_hidden_sizes=std_hidden_sizes,
                                  std_share_network=False,
                                  adaptive_std=True)
        model.build(self.input_var)
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
    def test_std_share_network_is_pickleable(self, output_dim, hidden_sizes):
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 5))
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  std_share_network=True,
                                  hidden_nonlinearity=None,
                                  hidden_w_init=tf.ones_initializer(),
                                  output_w_init=tf.ones_initializer())
        dist = model.build(input_var)

        # get output bias
        with tf.compat.v1.variable_scope('GaussianMLPModel2', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_std_network/output/bias')
        # assign it to all ones
        bias.load(tf.ones_like(bias).eval())

        output1 = self.sess.run(
            [dist.loc, tf.math.log(dist.stddev())],
            feed_dict={input_var: self.obs})

        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, None, 5))
            model_pickled = pickle.loads(h)
            dist2 = model_pickled.build(input_var)
            output2 = sess.run(
                [dist2.loc, tf.math.log(dist2.stddev())],
                feed_dict={input_var: self.obs})

            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_without_std_share_network_is_pickleable(self, output_dim,
                                                     hidden_sizes):
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 5))
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  std_share_network=False,
                                  adaptive_std=False,
                                  hidden_nonlinearity=None,
                                  hidden_w_init=tf.ones_initializer(),
                                  output_w_init=tf.ones_initializer())
        dist = model.build(input_var)

        # get output bias
        with tf.compat.v1.variable_scope('GaussianMLPModel2', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/output/bias')
        # assign it to all ones
        bias.load(tf.ones_like(bias).eval())

        output1 = self.sess.run(
            [dist.loc, tf.math.log(dist.stddev())],
            feed_dict={input_var: self.obs})

        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, None, 5))
            model_pickled = pickle.loads(h)
            dist2 = model_pickled.build(input_var)
            output2 = sess.run(
                [dist2.loc, tf.math.log(dist2.stddev())],
                feed_dict={input_var: self.obs})
            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('output_dim, hidden_sizes, std_hidden_sizes',
                             [(1, (0, ), (0, )), (1, (1, ), (1, )),
                              (1, (2, ), (2, )), (2, (3, ), (3, )),
                              (2, (1, 1), (1, 1)), (3, (2, 2), (2, 2))])
    def test_adaptive_std_is_pickleable(self, output_dim, hidden_sizes,
                                        std_hidden_sizes):
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 5))
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  std_hidden_sizes=std_hidden_sizes,
                                  std_share_network=False,
                                  adaptive_std=True,
                                  hidden_nonlinearity=None,
                                  hidden_w_init=tf.ones_initializer(),
                                  output_w_init=tf.ones_initializer(),
                                  std_hidden_nonlinearity=None,
                                  std_hidden_w_init=tf.ones_initializer(),
                                  std_output_w_init=tf.ones_initializer())
        dist = model.build(input_var)

        # get output bias
        with tf.compat.v1.variable_scope('GaussianMLPModel2', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/output/bias')
        # assign it to all ones
        bias.load(tf.ones_like(bias).eval())

        h = pickle.dumps(model)
        output1 = self.sess.run(
            [dist.loc, tf.math.log(dist.stddev())],
            feed_dict={input_var: self.obs})
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, None, 5))
            model_pickled = pickle.loads(h)
            dist2 = model_pickled.build(input_var)
            output2 = sess.run(
                [dist2.loc, tf.math.log(dist2.stddev())],
                feed_dict={input_var: self.obs})
            assert np.array_equal(output1, output2)

    # pylint: disable=assignment-from-no-return
    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_softplus_output_values(self, output_dim, hidden_sizes):
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  hidden_nonlinearity=None,
                                  std_share_network=False,
                                  adaptive_std=False,
                                  init_std=2,
                                  std_parameterization='softplus',
                                  hidden_w_init=tf.ones_initializer(),
                                  output_w_init=tf.ones_initializer())
        dist = model.build(self.input_var)

        mean, log_std = self.sess.run(
            [dist.loc, tf.math.log(dist.stddev())],
            feed_dict={self.input_var: self.obs})

        expected_mean = np.full([1, 1, output_dim], 5 * np.prod(hidden_sizes))
        expected_std_param = np.full([1, 1, output_dim], np.log(np.exp(2) - 1))
        expected_log_std = np.log(np.log(1. + np.exp(expected_std_param)))
        assert np.array_equal(mean, expected_mean)
        assert np.allclose(log_std, expected_log_std)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_exp_min_std(self, output_dim, hidden_sizes):
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  std_share_network=False,
                                  init_std=1,
                                  min_std=10,
                                  std_parameterization='exp')
        dist = model.build(self.input_var)

        log_std = self.sess.run(tf.math.log(dist.stddev()),
                                feed_dict={self.input_var: self.obs})

        expected_log_std = np.full([1, 1, output_dim], np.log(10))
        assert np.allclose(log_std, expected_log_std)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_exp_max_std(self, output_dim, hidden_sizes):
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  std_share_network=False,
                                  init_std=10,
                                  max_std=1,
                                  std_parameterization='exp')
        dist = model.build(self.input_var)

        log_std = self.sess.run(tf.math.log(dist.stddev()),
                                feed_dict={self.input_var: self.obs})

        expected_log_std = np.full([1, 1, output_dim], np.log(1))
        assert np.allclose(log_std, expected_log_std)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_softplus_min_std(self, output_dim, hidden_sizes):
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  std_share_network=False,
                                  init_std=1,
                                  min_std=10,
                                  std_parameterization='softplus')
        dist = model.build(self.input_var)

        log_std = self.sess.run(tf.math.log(dist.stddev()),
                                feed_dict={self.input_var: self.obs})

        expected_log_std = np.full([1, 1, output_dim], np.log(10))

        assert np.allclose(log_std, expected_log_std)

    @pytest.mark.parametrize('output_dim, hidden_sizes',
                             [(1, (0, )), (1, (1, )), (1, (2, )), (2, (3, )),
                              (2, (1, 1)), (3, (2, 2))])
    def test_softplus_max_std(self, output_dim, hidden_sizes):
        model = GaussianMLPModel2(output_dim=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  std_share_network=False,
                                  init_std=10,
                                  max_std=1,
                                  std_parameterization='softplus')
        dist = model.build(self.input_var)

        log_std = self.sess.run(tf.math.log(dist.stddev()),
                                feed_dict={self.input_var: self.obs})

        expected_log_std = np.full([1, 1, output_dim], np.log(1))

        # This test fails just outside of the default absolute tolerance.
        assert np.allclose(log_std, expected_log_std, atol=1e-7)

    def test_unknown_std_parameterization(self):
        with pytest.raises(ValueError):
            GaussianMLPModel2(output_dim=1, std_parameterization='unknown')
