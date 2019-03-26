"""
This script creates a test that fails when garage.tf.algos.DQN performance is
too low.
"""
import gym

from garage.experiment import LocalRunner
from garage.exploration_strategies import EpsilonGreedyStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteMLPQFunction
from tests.fixtures import TfGraphTestCase


class TestDQN(TfGraphTestCase):
    def test_dqn_cartpole(self):
        """Test DQN with CartPole environment."""
        with LocalRunner(self.sess) as runner:
            num_timesteps = 20000
            env = TfEnv(gym.make("CartPole-v0"))
            replay_buffer = SimpleReplayBuffer(
                env_spec=env.spec,
                size_in_transitions=int(5000),
                time_horizon=1)
            qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=(64, 64))
            policy = DiscreteQfDerivedPolicy(env_spec=env.spec, qf=qf)
            epilson_greedy_strategy = EpsilonGreedyStrategy(
                env_spec=env.spec,
                total_timesteps=num_timesteps,
                max_epsilon=1.0,
                min_epsilon=0.02,
                decay_ratio=0.1)
            algo = DQN(
                env_spec=env.spec,
                policy=policy,
                qf=qf,
                exploration_strategy=epilson_greedy_strategy,
                replay_buffer=replay_buffer,
                num_timesteps=num_timesteps,
                qf_lr=1e-4,
                max_path_length=1,
                discount=1.0,
                min_buffer_size=1e3,
                double_q=False,
                target_network_update_freq=500,
                buffer_batch_size=32)

            runner.setup(algo, env)
            last_avg_ret = runner.train(
                n_epochs=num_timesteps, n_epoch_cycles=1)
            assert last_avg_ret > 80

            env.close()
