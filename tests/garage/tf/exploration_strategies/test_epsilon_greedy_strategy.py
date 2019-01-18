"""
This script creates a unittest that tests Gaussian policies in
garage.tf.policies.
"""
import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.exploration_strategies import EpsilonGreedyStrategy
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteMLPQFunction
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestEpsilonGreedyStrategy(TfGraphTestCase):
    def test_epsilon_greedy_strategy(self):
        env = TfEnv(DummyDiscreteEnv())
        qf = DiscreteMLPQFunction(env.spec)
        policy = DiscreteQfDerivedPolicy(env_spec=env, qf=qf)

        epilson_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=env,
            total_step=100,
            max_epsilon=1.0,
            min_epsilon=0.02,
            decay_ratio=0.1)

        self.sess.run(tf.global_variables_initializer())

        env.reset()
        obs, _, _, _ = env.step(1)

        action = epilson_greedy_strategy.get_action(0, obs, policy)
        assert action in np.arange(env.action_space.n)
        actions = epilson_greedy_strategy.get_actions(0, [obs], policy)
        assert actions in np.arange(env.action_space.n)
