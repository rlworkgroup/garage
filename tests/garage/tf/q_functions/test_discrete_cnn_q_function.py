import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.q_functions import DiscreteCNNQFunction
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv
from tests.fixtures.models import SimpleCNNModel
from tests.fixtures.models import SimpleCNNModelWithMaxPooling
from tests.fixtures.models import SimpleMLPModel


class TestDiscreteCNNQFunction(TfGraphTestCase):
    def setup_method(self):
        super().setup_method()
        self.env = TfEnv(DummyDiscretePixelEnv())
        self.obs = self.env.reset()

    # yapf: disable
    @pytest.mark.parametrize('filter_dims, num_filters, strides', [
        ((3, ), (5, ), (1, )),
        ((3, ), (5, ), (2, )),
        ((3, 3), (5, 5), (1, 1)),
    ])
    # yapf: enable
    def test_get_action(self, filter_dims, num_filters, strides):
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_cnn_q_function.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.q_functions.'
                             'discrete_cnn_q_function.MLPModel'),
                            new=SimpleMLPModel):
                qf = DiscreteCNNQFunction(
                    env_spec=self.env.spec,
                    filter_dims=filter_dims,
                    num_filters=num_filters,
                    strides=strides,
                    dueling=False)

        action_dim = self.env.action_space.n
        expected_output = np.full(action_dim, 0.5)
        outputs = self.sess.run(qf.q_vals, feed_dict={qf.input: [self.obs]})
        assert np.array_equal(outputs[0], expected_output)
        outputs = self.sess.run(
            qf.q_vals, feed_dict={qf.input: [self.obs, self.obs, self.obs]})
        for output in outputs:
            assert np.array_equal(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('filter_dims, num_filters, strides', [
        ((3,), (5,), (1,)),
        ((3,), (5,), (2,)),
        ((3, 3), (5, 5), (1, 1)),
    ])
    # yapf: enable
    def test_get_action_dueling(self, filter_dims, num_filters, strides):
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_cnn_q_function.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.q_functions.'
                             'discrete_cnn_q_function.MLPDuelingModel'),
                            new=SimpleMLPModel):
                qf = DiscreteCNNQFunction(
                    env_spec=self.env.spec,
                    filter_dims=filter_dims,
                    num_filters=num_filters,
                    strides=strides,
                    dueling=True)

        action_dim = self.env.action_space.n
        expected_output = np.full(action_dim, 0.5)
        outputs = self.sess.run(qf.q_vals, feed_dict={qf.input: [self.obs]})
        assert np.array_equal(outputs[0], expected_output)
        outputs = self.sess.run(
            qf.q_vals, feed_dict={qf.input: [self.obs, self.obs, self.obs]})
        for output in outputs:
            assert np.array_equal(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('filter_dims, num_filters, strides, '
                             'pool_strides, pool_shapes', [
        ((3, ), (5, ), (1, ), (1, 1), (1, 1)),  # noqa: E122
        ((3, ), (5, ), (2, ), (2, 2), (2, 2)),
        ((3, 3), (5, 5), (1, 1), (1, 1), (1, 1)),
        ((3, 3), (5, 5), (1, 1), (2, 2), (2, 2))
    ])
    # yapf: enable
    def test_get_action_max_pooling(self, filter_dims, num_filters, strides,
                                    pool_strides, pool_shapes):
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_cnn_q_function.CNNModelWithMaxPooling'),
                        new=SimpleCNNModelWithMaxPooling):
            with mock.patch(('garage.tf.q_functions.'
                             'discrete_cnn_q_function.MLPModel'),
                            new=SimpleMLPModel):
                qf = DiscreteCNNQFunction(
                    env_spec=self.env.spec,
                    filter_dims=filter_dims,
                    num_filters=num_filters,
                    strides=strides,
                    max_pooling=True,
                    pool_strides=pool_strides,
                    pool_shapes=pool_shapes,
                    dueling=False)

        action_dim = self.env.action_space.n
        expected_output = np.full(action_dim, 0.5)
        outputs = self.sess.run(qf.q_vals, feed_dict={qf.input: [self.obs]})
        assert np.array_equal(outputs[0], expected_output)
        outputs = self.sess.run(
            qf.q_vals, feed_dict={qf.input: [self.obs, self.obs, self.obs]})
        for output in outputs:
            assert np.array_equal(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('filter_dims, num_filters, strides', [
        ((3, ), (5, ), (1, )),
        ((3, ), (5, ), (2, )),
        ((3, 3), (5, 5), (1, 1))
    ])
    # yapf: enable
    def test_get_qval_sym(self, filter_dims, num_filters, strides):
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_cnn_q_function.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.q_functions.'
                             'discrete_cnn_q_function.MLPModel'),
                            new=SimpleMLPModel):
                qf = DiscreteCNNQFunction(
                    env_spec=self.env.spec,
                    filter_dims=filter_dims,
                    num_filters=num_filters,
                    strides=strides,
                    dueling=False)
        output1 = self.sess.run(qf.q_vals, feed_dict={qf.input: [self.obs]})

        obs_dim = self.env.observation_space.shape
        action_dim = self.env.action_space.n

        input_var = tf.compat.v1.placeholder(
            tf.float32, shape=(None, ) + obs_dim)
        q_vals = qf.get_qval_sym(input_var, 'another')
        output2 = self.sess.run(q_vals, feed_dict={input_var: [self.obs]})

        expected_output = np.full(action_dim, 0.5)

        assert np.array_equal(output1, output2)
        assert np.array_equal(output2[0], expected_output)

    # yapf: disable
    @pytest.mark.parametrize('filter_dims, num_filters, strides', [
        ((3, ), (5, ), (1, )),
        ((3, ), (5, ), (2, )),
        ((3, 3), (5, 5), (1, 1)),
    ])
    # yapf: enable
    def test_is_pickleable(self, filter_dims, num_filters, strides):
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_cnn_q_function.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.q_functions.'
                             'discrete_cnn_q_function.MLPModel'),
                            new=SimpleMLPModel):
                qf = DiscreteCNNQFunction(
                    env_spec=self.env.spec,
                    filter_dims=filter_dims,
                    num_filters=num_filters,
                    strides=strides,
                    dueling=False)
        with tf.compat.v1.variable_scope(
                'DiscreteCNNQFunction/Sequential/SimpleMLPModel', reuse=True):
            return_var = tf.compat.v1.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())

        output1 = self.sess.run(qf.q_vals, feed_dict={qf.input: [self.obs]})

        h_data = pickle.dumps(qf)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            qf_pickled = pickle.loads(h_data)
            output2 = sess.run(
                qf_pickled.q_vals, feed_dict={qf_pickled.input: [self.obs]})

        assert np.array_equal(output1, output2)

    # yapf: disable
    @pytest.mark.parametrize('filter_dims, num_filters, strides', [
        ((3, ), (5, ), (1, )),
        ((3, ), (5, ), (2, )),
        ((3, 3), (5, 5), (1, 1))
    ])
    # yapf: enable
    def test_clone(self, filter_dims, num_filters, strides):
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_cnn_q_function.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.q_functions.'
                             'discrete_cnn_q_function.MLPModel'),
                            new=SimpleMLPModel):
                qf = DiscreteCNNQFunction(
                    env_spec=self.env.spec,
                    filter_dims=filter_dims,
                    num_filters=num_filters,
                    strides=strides,
                    dueling=False)
        qf_clone = qf.clone('another_qf')
        assert qf_clone._filter_dims == qf._filter_dims
        assert qf_clone._num_filters == qf._num_filters
        assert qf_clone._strides == qf._strides
