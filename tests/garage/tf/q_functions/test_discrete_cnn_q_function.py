import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.tf.models import CNNModel
from garage.tf.q_functions import DiscreteCNNQFunction

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv, DummyDiscretePixelEnv
from tests.fixtures.models import (SimpleCNNModel,
                                   SimpleCNNModelWithMaxPooling,
                                   SimpleMLPModel)


class TestDiscreteCNNQFunction(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.env = GymEnv(DummyDiscretePixelEnv())
        self.obs = self.env.reset()[0]

    # yapf: disable
    @pytest.mark.parametrize('filters, strides', [
        (((5, (3, 3)), ), (1, )),
        (((5, (3, 3)), ), (2, )),
        (((5, (3, 3)), (5, (3, 3))), (1, 1)),
    ])
    # yapf: enable
    def test_get_action(self, filters, strides):
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_cnn_q_function.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.q_functions.'
                             'discrete_cnn_q_function.MLPModel'),
                            new=SimpleMLPModel):
                qf = DiscreteCNNQFunction(env_spec=self.env.spec,
                                          filters=filters,
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

    @pytest.mark.parametrize('obs_dim', [[1], [2], [1, 1, 1, 1], [2, 2, 2, 2]])
    def test_invalid_obs_shape(self, obs_dim):
        boxEnv = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim))
        with pytest.raises(ValueError):
            DiscreteCNNQFunction(env_spec=boxEnv.spec,
                                 filters=((5, (3, 3)), ),
                                 strides=(2, ),
                                 dueling=False)

    def test_obs_is_image(self):
        image_env = GymEnv(DummyDiscretePixelEnv(), is_image=True)
        with mock.patch(('garage.tf.models.'
                         'categorical_cnn_model.CNNModel._build'),
                        autospec=True,
                        side_effect=CNNModel._build) as build:

            qf = DiscreteCNNQFunction(env_spec=image_env.spec,
                                      filters=((5, (3, 3)), ),
                                      strides=(2, ),
                                      dueling=False)
            normalized_obs = build.call_args_list[0][0][1]

            input_ph = qf.input
            assert input_ph != normalized_obs

            fake_obs = [np.full(image_env.spec.observation_space.shape, 255)]
            assert (self.sess.run(normalized_obs,
                                  feed_dict={input_ph: fake_obs}) == 1.).all()

            obs_dim = image_env.spec.observation_space.shape
            state_input = tf.compat.v1.placeholder(tf.uint8,
                                                   shape=(None, ) + obs_dim)

            qf.build(state_input, name='another')
            normalized_obs = build.call_args_list[1][0][1]

            fake_obs = [np.full(image_env.spec.observation_space.shape, 255)]
            assert (self.sess.run(normalized_obs,
                                  feed_dict={state_input:
                                             fake_obs}) == 1.).all()

    def test_obs_not_image(self):
        env = self.env
        with mock.patch(('garage.tf.models.'
                         'categorical_cnn_model.CNNModel._build'),
                        autospec=True,
                        side_effect=CNNModel._build) as build:

            qf = DiscreteCNNQFunction(env_spec=env.spec,
                                      filters=((5, (3, 3)), ),
                                      strides=(2, ),
                                      dueling=False)
            normalized_obs = build.call_args_list[0][0][1]

            input_ph = qf.input
            assert input_ph == normalized_obs

            fake_obs = [np.full(env.spec.observation_space.shape, 255)]
            assert (self.sess.run(normalized_obs,
                                  feed_dict={input_ph:
                                             fake_obs}) == 255.).all()

            obs_dim = env.spec.observation_space.shape
            state_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, ) + obs_dim)

            qf.build(state_input, name='another')
            normalized_obs = build.call_args_list[1][0][1]

            fake_obs = [np.full(env.spec.observation_space.shape, 255)]
            assert (self.sess.run(normalized_obs,
                                  feed_dict={state_input:
                                             fake_obs}) == 255).all()

    # yapf: disable
    @pytest.mark.parametrize('filters, strides', [
        (((5, (3, 3)), ), (1, )),
        (((5, (3, 3)), ), (2, )),
        (((5, (3, 3)), (5, (3, 3))), (1, 1)),
    ])
    # yapf: enable
    def test_get_action_dueling(self, filters, strides):
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_cnn_q_function.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.q_functions.'
                             'discrete_cnn_q_function.MLPDuelingModel'),
                            new=SimpleMLPModel):
                qf = DiscreteCNNQFunction(env_spec=self.env.spec,
                                          filters=filters,
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
    @pytest.mark.parametrize('filters, strides, pool_strides, pool_shapes', [
        (((5, (3, 3)), ), (1, ), (1, 1), (1, 1)),  # noqa: E122
        (((5, (3, 3)), ), (2, ), (2, 2), (2, 2)),  # noqa: E122
        (((5, (3, 3)), (5, (3, 3))), (1, 1), (1, 1), (1, 1)),  # noqa: E122
        (((5, (3, 3)), (5, (3, 3))), (1, 1), (2, 2), (2, 2))  # noqa: E122
    ])  # noqa: E122
    # yapf: enable
    def test_get_action_max_pooling(self, filters, strides, pool_strides,
                                    pool_shapes):
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_cnn_q_function.CNNModelWithMaxPooling'),
                        new=SimpleCNNModelWithMaxPooling):
            with mock.patch(('garage.tf.q_functions.'
                             'discrete_cnn_q_function.MLPModel'),
                            new=SimpleMLPModel):
                qf = DiscreteCNNQFunction(env_spec=self.env.spec,
                                          filters=filters,
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
    @pytest.mark.parametrize('filters, strides', [
        (((5, (3, 3)), ), (1, )),
        (((5, (3, 3)), ), (2, )),
        (((5, (3, 3)), (5, (3, 3))), (1, 1)),
    ])
    # yapf: enable
    def test_build(self, filters, strides):
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_cnn_q_function.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.q_functions.'
                             'discrete_cnn_q_function.MLPModel'),
                            new=SimpleMLPModel):
                qf = DiscreteCNNQFunction(env_spec=self.env.spec,
                                          filters=filters,
                                          strides=strides,
                                          dueling=False)
        output1 = self.sess.run(qf.q_vals, feed_dict={qf.input: [self.obs]})

        obs_dim = self.env.observation_space.shape
        action_dim = self.env.action_space.n

        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, ) + obs_dim)
        q_vals = qf.build(input_var, 'another')
        output2 = self.sess.run(q_vals, feed_dict={input_var: [self.obs]})

        expected_output = np.full(action_dim, 0.5)

        assert np.array_equal(output1, output2)
        assert np.array_equal(output2[0], expected_output)

    # yapf: disable
    @pytest.mark.parametrize('filters, strides', [
        (((5, (3, 3)), ), (1, )),
        (((5, (3, 3)), ), (2, )),
        (((5, (3, 3)), (5, (3, 3))), (1, 1)),
    ])
    # yapf: enable
    def test_is_pickleable(self, filters, strides):
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_cnn_q_function.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.q_functions.'
                             'discrete_cnn_q_function.MLPModel'),
                            new=SimpleMLPModel):
                qf = DiscreteCNNQFunction(env_spec=self.env.spec,
                                          filters=filters,
                                          strides=strides,
                                          dueling=False)
        with tf.compat.v1.variable_scope('DiscreteCNNQFunction/SimpleMLPModel',
                                         reuse=True):
            return_var = tf.compat.v1.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())

        output1 = self.sess.run(qf.q_vals, feed_dict={qf.input: [self.obs]})

        h_data = pickle.dumps(qf)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            qf_pickled = pickle.loads(h_data)
            output2 = sess.run(qf_pickled.q_vals,
                               feed_dict={qf_pickled.input: [self.obs]})

        assert np.array_equal(output1, output2)

    # yapf: disable
    @pytest.mark.parametrize('filters, strides', [
        (((5, (3, 3)), ), (1, )),
        (((5, (3, 3)), ), (2, )),
        (((5, (3, 3)), (5, (3, 3))), (1, 1)),
    ])
    # yapf: enable
    def test_clone(self, filters, strides):
        qf = DiscreteCNNQFunction(env_spec=self.env.spec,
                                  filters=filters,
                                  strides=strides,
                                  dueling=False)
        qf_clone = qf.clone('another_qf')
        assert qf_clone._filters == qf._filters
        assert qf_clone._strides == qf._strides
        for cloned_param, param in zip(qf_clone.parameters.values(),
                                       qf.parameters.values()):
            assert np.array_equal(cloned_param, param)
