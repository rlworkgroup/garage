"""
This script creates a unittest that tests Gaussian policies in
garage.tf.policies.
"""
from unittest import mock

import numpy as np
import tensorflow as tf

from garage.tf.core.mlp import mlp
from garage.tf.envs import TfEnv
from garage.tf.policies import DiscreteQfDerivedPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestQfDerivedPolicy(TfGraphTestCase):
    def test_discrete_qf_derived_policy(self):
        # mock a q_function
        env = TfEnv(DummyDiscreteEnv())
        obs_ph = tf.placeholder(
            tf.float32, shape=(None, ) + env.observation_space.shape)
        qf_function = mock.Mock()
        qf_function.q_val = mlp(
            input_var=obs_ph,
            output_dim=env.action_space.flat_dim,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            name="mlp")
        qf_function.obs_ph = obs_ph

        policy = DiscreteQfDerivedPolicy(env_spec=env, qf=qf_function)

        self.sess.run(tf.global_variables_initializer())

        obs, _, _, _ = env.step(1)
        action = policy.get_action(obs)
        assert action in np.arange(env.action_space.n)
        actions = policy.get_actions([obs])
        assert actions in np.arange(env.action_space.n)
