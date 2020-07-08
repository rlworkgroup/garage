import pickle

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from garage.tf.models import CategoricalCNNModel

from tests.fixtures import TfGraphTestCase


class TestCategoricalMLPModel(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        batch_size = 5
        input_width = 10
        input_height = 10
        self._obs_input = np.ones(
            (batch_size, 1, input_width, input_height, 3))
        self._input_shape = (input_width, input_height, 3
                             )  # height, width, channel
        self._input_ph = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, None) +
                                                  self._input_shape,
                                                  name='input')

    def test_dist(self):
        model = CategoricalCNNModel(output_dim=1,
                                    filters=((5, (3, 3)), ),
                                    strides=(1, ),
                                    padding='VALID')
        dist = model.build(self._input_ph).dist
        assert isinstance(dist, tfp.distributions.OneHotCategorical)

    def test_instantiate_with_different_name(self):
        model = CategoricalCNNModel(output_dim=1,
                                    filters=((5, (3, 3)), ),
                                    strides=(1, ),
                                    padding='VALID')
        model.build(self._input_ph)
        model.build(self._input_ph, name='another_model')

    # yapf: disable
    @pytest.mark.parametrize(
        'output_dim, filters, strides, padding, hidden_sizes', [
            (1, ((1, (1, 1)), ), (1, ), 'SAME', (1, )),
            (1, ((3, (3, 3)), ), (2, ), 'VALID', (2, )),
            (1, ((3, (3, 3)), ), (2, ), 'SAME', (3, )),
            (2, ((3, (3, 3)), (32, (3, 3))), (2, 2), 'VALID', (1, 1)),
            (3, ((3, (3, 3)), (32, (3, 3))), (2, 2), 'SAME', (2, 2)),
        ])
    # yapf: enable
    def test_is_pickleable(self, output_dim, filters, strides, padding,
                           hidden_sizes):
        model = CategoricalCNNModel(output_dim=output_dim,
                                    filters=filters,
                                    strides=strides,
                                    padding=padding,
                                    hidden_sizes=hidden_sizes,
                                    hidden_nonlinearity=None,
                                    hidden_w_init=tf.ones_initializer(),
                                    output_w_init=tf.ones_initializer())
        dist = model.build(self._input_ph).dist
        # assign bias to all one
        with tf.compat.v1.variable_scope('CategoricalCNNModel', reuse=True):
            cnn_bias = tf.compat.v1.get_variable('CNNModel/cnn/h0/bias')
            bias = tf.compat.v1.get_variable('MLPModel/mlp/hidden_0/bias')

        bias.load(tf.ones_like(bias).eval())
        cnn_bias.load(tf.ones_like(cnn_bias).eval())

        output1 = self.sess.run(dist.probs,
                                feed_dict={self._input_ph: self._obs_input})

        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, None) +
                                                 self._input_shape)
            model_pickled = pickle.loads(h)
            dist2 = model_pickled.build(input_var).dist
            output2 = sess.run(dist2.probs,
                               feed_dict={input_var: self._obs_input})

            assert np.array_equal(output1, output2)
