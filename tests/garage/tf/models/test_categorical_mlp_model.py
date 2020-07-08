import pickle

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from garage.tf.models import CategoricalMLPModel

from tests.fixtures import TfGraphTestCase


class TestCategoricalMLPModel(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
        self.obs = np.ones((1, 5))

    def test_dist(self):
        model = CategoricalMLPModel(output_dim=1)
        dist = model.build(self.input_var).dist
        assert isinstance(dist, tfp.distributions.OneHotCategorical)

    @pytest.mark.parametrize('output_dim', [1, 2, 5, 10])
    def test_output_normalized(self, output_dim):
        model = CategoricalMLPModel(output_dim=output_dim)
        obs_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, output_dim))
        obs = np.ones((1, output_dim))
        dist = model.build(obs_ph).dist
        probs = tf.compat.v1.get_default_session().run(tf.reduce_sum(
            dist.probs),
                                                       feed_dict={obs_ph: obs})
        assert np.isclose(probs, 1.0)

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_sizes', [
        (1, (1, )),
        (1, (2, )),
        (2, (3, )),
        (2, (1, 1)),
        (3, (2, 2)),
    ])
    # yapf: enable
    def test_is_pickleable(self, output_dim, hidden_sizes):
        model = CategoricalMLPModel(output_dim=output_dim,
                                    hidden_sizes=hidden_sizes,
                                    hidden_nonlinearity=None,
                                    hidden_w_init=tf.ones_initializer(),
                                    output_w_init=tf.ones_initializer())
        dist = model.build(self.input_var).dist
        # assign bias to all one
        with tf.compat.v1.variable_scope('CategoricalMLPModel/mlp',
                                         reuse=True):
            bias = tf.compat.v1.get_variable('hidden_0/bias')

        bias.load(tf.ones_like(bias).eval())

        output1 = self.sess.run(dist.probs,
                                feed_dict={self.input_var: self.obs})

        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(h)
            dist2 = model_pickled.build(input_var).dist
            output2 = sess.run(dist2.probs, feed_dict={input_var: self.obs})

            assert np.array_equal(output1, output2)
