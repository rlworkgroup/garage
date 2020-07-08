import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.models import MLPDuelingModel, MLPMergeModel, MLPModel

from tests.fixtures import TfGraphTestCase


class TestMLPModel(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
        self.obs = np.ones((1, 5))

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_sizes', [
        (1, (0, )),
        (1, (1, )),
        (1, (2, )),
        (2, (3, )),
        (2, (1, 1)),
        (3, (2, 2)),
    ])
    # yapf: enable
    def test_output_values(self, output_dim, hidden_sizes):
        model = MLPModel(output_dim=output_dim,
                         hidden_sizes=hidden_sizes,
                         hidden_nonlinearity=None,
                         hidden_w_init=tf.ones_initializer(),
                         output_w_init=tf.ones_initializer())
        outputs = model.build(self.input_var).outputs
        output = self.sess.run(outputs, feed_dict={self.input_var: self.obs})

        expected_output = np.full([1, output_dim], 5 * np.prod(hidden_sizes))

        assert np.array_equal(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_sizes', [
        (1, (0, )),
        (1, (1, )),
        (1, (2, )),
        (2, (3, )),
        (2, (1, 1)),
        (3, (2, 2)),
    ])
    # yapf: enable
    def test_output_values_dueling(self, output_dim, hidden_sizes):
        model = MLPDuelingModel(output_dim=output_dim,
                                hidden_sizes=hidden_sizes,
                                hidden_nonlinearity=None,
                                hidden_w_init=tf.ones_initializer(),
                                output_w_init=tf.ones_initializer())
        outputs = model.build(self.input_var).outputs
        output = self.sess.run(outputs, feed_dict={self.input_var: self.obs})

        expected_output = np.full([1, output_dim], 5 * np.prod(hidden_sizes))

        assert np.array_equal(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_sizes', [
        (1, (0, )),
        (1, (1, )),
        (1, (2, )),
        (2, (3, )),
        (2, (1, 1)),
        (3, (2, 2)),
    ])
    # yapf: enable
    def test_output_values_merging(self, output_dim, hidden_sizes):
        model = MLPMergeModel(output_dim=output_dim,
                              hidden_sizes=hidden_sizes,
                              concat_layer=0,
                              hidden_nonlinearity=None,
                              hidden_w_init=tf.ones_initializer(),
                              output_w_init=tf.ones_initializer())

        input_var2 = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
        obs2 = np.ones((1, 5))

        outputs = model.build(self.input_var, input_var2).outputs
        output = self.sess.run(outputs,
                               feed_dict={
                                   self.input_var: self.obs,
                                   input_var2: obs2
                               })

        expected_output = np.full([1, output_dim], 10 * np.prod(hidden_sizes))
        assert np.array_equal(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('output_dim, hidden_sizes', [
        (1, (0, )),
        (1, (1, )),
        (1, (2, )),
        (2, (3, )),
        (2, (1, 1)),
        (3, (2, 2)),
    ])
    # yapf: enable
    def test_is_pickleable(self, output_dim, hidden_sizes):
        model = MLPModel(output_dim=output_dim,
                         hidden_sizes=hidden_sizes,
                         hidden_nonlinearity=None,
                         hidden_w_init=tf.ones_initializer(),
                         output_w_init=tf.ones_initializer())
        outputs = model.build(self.input_var).outputs

        # assign bias to all one
        with tf.compat.v1.variable_scope('MLPModel/mlp', reuse=True):
            bias = tf.compat.v1.get_variable('hidden_0/bias')

        bias.load(tf.ones_like(bias).eval())

        output1 = self.sess.run(outputs, feed_dict={self.input_var: self.obs})

        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(h)
            outputs = model_pickled.build(input_var).outputs
            output2 = sess.run(outputs, feed_dict={input_var: self.obs})

            assert np.array_equal(output1, output2)
