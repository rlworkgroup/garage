import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.tf.q_functions import ContinuousMLPQFunction

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestContinuousMLPQFunction(TfGraphTestCase):

    @pytest.mark.parametrize('hidden_sizes', [(1, ), (2, ), (3, ), (1, 1),
                                              (2, 2)])
    def test_q_vals(self, hidden_sizes):
        env = GymEnv(DummyBoxEnv())
        obs_dim = env.spec.observation_space.flat_dim
        act_dim = env.spec.action_space.flat_dim
        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    action_merge_layer=0,
                                    hidden_sizes=hidden_sizes,
                                    hidden_nonlinearity=None,
                                    hidden_w_init=tf.ones_initializer(),
                                    output_w_init=tf.ones_initializer())
        obs = np.full(obs_dim, 1).flatten()
        act = np.full(act_dim, 1).flatten()

        expected_output = np.full((1, ),
                                  (obs_dim + act_dim) * np.prod(hidden_sizes))
        outputs = qf.get_qval([obs], [act])
        assert np.array_equal(outputs[0], expected_output)

        outputs = qf.get_qval([obs, obs, obs], [act, act, act])
        for output in outputs:
            assert np.array_equal(output, expected_output)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((2, ), (2, )),
        ((1, 1), (1, )),
        ((2, 2), (2, )),
    ])
    def test_output_shape(self, obs_dim, action_dim):
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        qf = ContinuousMLPQFunction(env_spec=env.spec)
        env.reset()
        obs = env.step(1).observation
        obs = obs.flatten()
        act = np.full(action_dim, 0.5).flatten()

        outputs = qf.get_qval([obs], [act])
        assert outputs.shape == (1, 1)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((2, ), (2, )),
        ((1, 1), (1, )),
        ((2, 2), (2, )),
    ])
    def test_build(self, obs_dim, action_dim):
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    action_merge_layer=0,
                                    hidden_sizes=(1, ),
                                    hidden_nonlinearity=None,
                                    hidden_w_init=tf.ones_initializer(),
                                    output_w_init=tf.ones_initializer())
        obs = np.full(obs_dim, 1).flatten()
        act = np.full(action_dim, 1).flatten()

        output1 = qf.get_qval([obs], [act])

        input_var1 = tf.compat.v1.placeholder(tf.float32,
                                              shape=(None, obs.shape[0]))
        input_var2 = tf.compat.v1.placeholder(tf.float32,
                                              shape=(None, act.shape[0]))
        q_vals = qf.build(input_var1, input_var2, 'another')
        output2 = self.sess.run(q_vals,
                                feed_dict={
                                    input_var1: [obs],
                                    input_var2: [act]
                                })

        expected_output = np.full((1, ),
                                  np.prod(obs_dim) + np.prod(action_dim))

        assert np.array_equal(output1, output2)
        assert np.array_equal(output2[0], expected_output)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((2, ), (2, )),
        ((1, 1), (1, )),
        ((2, 2), (2, )),
    ])
    def test_is_pickleable(self, obs_dim, action_dim):
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        qf = ContinuousMLPQFunction(env_spec=env.spec)
        env.reset()
        obs = env.step(1).observation
        obs = obs.flatten()
        act = np.full(action_dim, 0.5).flatten()

        with tf.compat.v1.variable_scope('ContinuousMLPQFunction', reuse=True):
            bias = tf.compat.v1.get_variable('mlp_concat/hidden_0/bias')
        # assign it to all one
        bias.load(tf.ones_like(bias).eval())

        output1 = qf.get_qval([obs], [act])

        h_data = pickle.dumps(qf)
        with tf.compat.v1.Session(graph=tf.Graph()):
            qf_pickled = pickle.loads(h_data)
            output2 = qf_pickled.get_qval([obs], [act])

        assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_sizes', [
        ((1, ), (1, ), (3, )),
        ((2, ), (2, ), (32, )),
        ((1, 1), (1, ), (3, 3)),
        ((2, 2), (2, ), (32, 32)),
    ])
    def test_clone(self, obs_dim, action_dim, hidden_sizes):
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=hidden_sizes)
        qf_clone = qf.clone('another_qf')
        assert qf_clone._hidden_sizes == qf._hidden_sizes
        for cloned_param, param in zip(qf_clone.parameters.values(),
                                       qf.parameters.values()):
            assert np.array_equal(cloned_param, param)
