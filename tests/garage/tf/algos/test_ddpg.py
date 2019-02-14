"""
This script creates a test that fails when garage.tf.algos.DDPG performance is
too low.
"""
import gym
import tensorflow as tf

import garage.misc.logger as logger
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from tests.fixtures import TfGraphTestCase


class TestDDPG(TfGraphTestCase):
    def test_ddpg_pendulum(self):
        """Test PPO with Pendulum environment."""
        logger.reset()
        env = TfEnv(gym.make('InvertedDoublePendulum-v2'))
        action_noise = OUStrategy(env.spec, sigma=0.2)
        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)
        qf = ContinuousMLPQFunction(
            env_spec=env.spec,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=tf.nn.relu)
        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)
        algo = DDPG(
            env,
            policy=policy,
            policy_lr=1e-4,
            qf_lr=1e-3,
            qf=qf,
            replay_buffer=replay_buffer,
            plot=False,
            target_update_tau=1e-2,
            n_epochs=10,
            n_epoch_cycles=20,
            max_path_length=100,
            n_train_steps=50,
            discount=0.9,
            min_buffer_size=int(1e4),
            exploration_strategy=action_noise,
        )
        last_avg_ret = algo.train(sess=self.sess)
        assert last_avg_ret > 60

        env.close()
