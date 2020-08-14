import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GarageEnv
from garage.tf.policies import GaussianMLPPolicy

# yapf: disable
from tests.fixtures import TfGraphTestCase  # noqa: I202
from tests.fixtures.envs.dummy import (DummyBoxEnv,
                                       DummyDictEnv,
                                       DummyDiscreteEnv)

# yapf: enable


class TestGaussianMLPPolicy(TfGraphTestCase):

    def test_invalid_env(self):
        env = GarageEnv(DummyDiscreteEnv())
        with pytest.raises(ValueError):
            GaussianMLPPolicy(env_spec=env.spec)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_get_action(self, obs_dim, action_dim):
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = GaussianMLPPolicy(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        action, _ = policy.get_action(obs.flatten())
        assert env.action_space.contains(action)
        actions, _ = policy.get_actions(
            [obs.flatten(), obs.flatten(),
             obs.flatten()])
        for action in actions:
            assert env.action_space.contains(action)

    def test_get_action_dict_space(self):
        env = GarageEnv(
            DummyDictEnv(obs_space_type='box', act_space_type='box'))
        policy = GaussianMLPPolicy(env_spec=env.spec)
        obs = env.reset()

        action, _ = policy.get_action(obs)
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs, obs])
        for action in actions:
            assert env.action_space.contains(action)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_build(self, obs_dim, action_dim):
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = GaussianMLPPolicy(env_spec=env.spec)
        obs = env.reset()

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist
        output1 = self.sess.run(
            [policy.distribution.loc],
            feed_dict={policy.model.input: [[obs.flatten()]]})
        output2 = self.sess.run([dist_sym.loc],
                                feed_dict={state_input: [[obs.flatten()]]})
        assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_is_pickleable(self, obs_dim, action_dim):
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = GaussianMLPPolicy(env_spec=env.spec)

        obs = env.reset()

        with tf.compat.v1.variable_scope('GaussianMLPPolicy/GaussianMLPModel',
                                         reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/hidden_0/bias')
        # assign it to all one
        bias.load(tf.ones_like(bias).eval())
        output1 = self.sess.run(
            [policy.distribution.loc,
             policy.distribution.stddev()],
            feed_dict={policy.model.input: [[obs.flatten()]]})

        p = pickle.dumps(policy)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            output2 = sess.run(
                [
                    policy_pickled.distribution.loc,
                    policy_pickled.distribution.stddev()
                ],
                feed_dict={policy_pickled.model.input: [[obs.flatten()]]})
            assert np.array_equal(output2, output1)
