import pickle

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from garage.tf.models import CategoricalGRUModel
from tests.fixtures import TfGraphTestCase


class TestCategoricalGRUModel(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.batch_size = 1
        self.time_step = 1
        self.feature_shape = 2
        self.output_dim = 1

        self.obs_inputs = np.full(
            (self.batch_size, self.time_step, self.feature_shape), 1.)
        self.obs_input = np.full((self.batch_size, self.feature_shape), 1.)

        self.input_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, None,
                                                         self.feature_shape),
                                                  name='input')
        self.step_input_var = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.feature_shape), name='input')

    def test_dist(self):
        model = CategoricalGRUModel(output_dim=1, hidden_dim=1)
        step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size, 1),
                                                   name='step_hidden',
                                                   dtype=tf.float32)
        model.build(self.input_var, self.step_input_var, step_hidden_var)
        assert isinstance(model.networks['default'].dist,
                          tfp.distributions.OneHotCategorical)

    def test_output_nonlinearity(self):
        model = CategoricalGRUModel(output_dim=1,
                                    hidden_dim=4,
                                    output_nonlinearity=lambda x: x / 2)
        obs_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 1))
        step_obs_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        step_hidden_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
        obs = np.ones((1, 1, 1))
        dist, _, _, _ = model.build(obs_ph, step_obs_ph, step_hidden_ph)
        probs = tf.compat.v1.get_default_session().run(dist.probs,
                                                       feed_dict={obs_ph: obs})
        assert probs == [0.5]

    @pytest.mark.parametrize('output_dim', [1, 2, 5, 10])
    def test_output_normalized(self, output_dim):
        model = CategoricalGRUModel(output_dim=output_dim, hidden_dim=4)
        obs_ph = tf.compat.v1.placeholder(tf.float32,
                                          shape=(None, None, output_dim))
        step_obs_ph = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, output_dim))
        step_hidden_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
        obs = np.ones((1, 1, output_dim))
        dist, _, _, _ = model.build(obs_ph, step_obs_ph, step_hidden_ph)
        probs = tf.compat.v1.get_default_session().run(tf.reduce_sum(
            dist.probs),
                                                       feed_dict={obs_ph: obs})
        assert np.isclose(probs, 1.0)

    def test_is_pickleable(self):
        model = CategoricalGRUModel(output_dim=1, hidden_dim=1)
        step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size, 1),
                                                   name='step_hidden',
                                                   dtype=tf.float32)
        model.build(self.input_var, self.step_input_var, step_hidden_var)

        # assign bias to all one
        with tf.compat.v1.variable_scope('CategoricalGRUModel/gru',
                                         reuse=True):
            init_hidden = tf.compat.v1.get_variable('initial_hidden')

        init_hidden.load(tf.ones_like(init_hidden).eval())

        hidden = np.zeros((self.batch_size, 1))

        outputs1 = self.sess.run(model.networks['default'].dist.probs,
                                 feed_dict={self.input_var: self.obs_inputs})
        output1 = self.sess.run(
            [
                model.networks['default'].step_output,
                model.networks['default'].step_hidden
            ],
            # yapf: disable
            feed_dict={
                self.step_input_var: self.obs_input,
                step_hidden_var: hidden
            })
        # yapf: enable
        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(h)

            input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, None,
                                                        self.feature_shape),
                                                 name='input')
            step_input_var = tf.compat.v1.placeholder(
                tf.float32, shape=(None, self.feature_shape), name='input')
            step_hidden_var = tf.compat.v1.placeholder(shape=(self.batch_size,
                                                              1),
                                                       name='initial_hidden',
                                                       dtype=tf.float32)

            model_pickled.build(input_var, step_input_var, step_hidden_var)
            outputs2 = sess.run(model_pickled.networks['default'].dist.probs,
                                feed_dict={input_var: self.obs_inputs})
            output2 = sess.run(
                [
                    model_pickled.networks['default'].step_output,
                    model_pickled.networks['default'].step_hidden
                ],
                # yapf: disable
                feed_dict={
                    step_input_var: self.obs_input,
                    step_hidden_var: hidden
                })
            # yapf: enable
            assert np.array_equal(outputs1, outputs2)
            assert np.array_equal(output1, output2)
