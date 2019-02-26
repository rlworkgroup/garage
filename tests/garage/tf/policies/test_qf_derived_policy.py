"""
This script creates a unittest that tests Gaussian policies in
garage.tf.policies.
"""
import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import DiscreteQfDerivedPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class SimpleQFunction:
    """Simple QFunction for testing."""

    def __init__(self, env_spec):
        obs_dim = env_spec.observation_space.shape

        self.obs_ph = tf.placeholder(
            tf.float32, (None, ) + obs_dim, name="obs")
        self.q_val = self.build_net(self.obs_ph)

    def build_net(self, input_var):
        return tf.identity(input_var)


class TestQfDerivedPolicy(TfGraphTestCase):
    def test_discrete_qf_derived_policy(self):
        env = TfEnv(DummyDiscreteEnv())
        qf = SimpleQFunction(env.spec)
        policy = DiscreteQfDerivedPolicy(env_spec=env, qf=qf)

        self.sess.run(tf.global_variables_initializer())

        env.reset()
        obs, _, _, _ = env.step(1)
        action = policy.get_action(obs)
        assert action in np.arange(env.action_space.n)
        actions = policy.get_actions([obs])
        assert actions in np.arange(env.action_space.n)
