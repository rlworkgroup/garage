"""
This script creates a unittest that tests Gaussian policies in
garage.tf.policies.
"""
import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteMLPQFunction
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestQfDerivedPolicy(TfGraphTestCase):
    def test_discrete_qf_derived_policy(self):
        env = TfEnv(DummyDiscreteEnv())
        qf = DiscreteMLPQFunction(env.spec)
        policy = DiscreteQfDerivedPolicy(env_spec=env, qf=qf)

        self.sess.run(tf.global_variables_initializer())

        env.reset()
        obs, _, _, _ = env.step(1)
        action = policy.get_action(obs)
        assert action in np.arange(env.action_space.n)
        actions = policy.get_actions([obs])
        assert actions in np.arange(env.action_space.n)
