import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GarageEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.envs.dummy import DummyDiscreteEnv


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
        obs_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, None, env.observation_space.flat_dim],
            name='obs')
        policy = GaussianMLPPolicy(env_spec=env.spec)

        policy.build(obs_var)
        env.reset()
        obs, _, _, _ = env.step(1)

        action, _ = policy.get_action(obs.flatten())
        assert env.action_space.contains(action)
        actions, _ = policy.get_actions(
            [obs.flatten(), obs.flatten(),
             obs.flatten()])
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
    def test_is_pickleable(self, obs_dim, action_dim):
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        obs_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, None, env.observation_space.flat_dim],
            name='obs')
        policy = GaussianMLPPolicy(env_spec=env.spec)

        policy.build(obs_var)
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
            obs_var = tf.compat.v1.placeholder(
                tf.float32,
                shape=[None, None, env.observation_space.flat_dim],
                name='obs')
            policy_pickled = pickle.loads(p)
            policy_pickled.build(obs_var)
            output2 = sess.run(
                [
                    policy_pickled.distribution.loc,
                    policy_pickled.distribution.stddev()
                ],
                feed_dict={policy_pickled.model.input: [[obs.flatten()]]})
            assert np.array_equal(output1, output2)

    def test_clone(self):
        env = GarageEnv(DummyBoxEnv(obs_dim=(10, ), action_dim=(4, )))
        policy = GaussianMLPPolicy(env_spec=env.spec)
        policy_clone = policy.clone('GaussnaMLPPolicyClone')
        assert policy.env_spec == policy_clone.env_spec
