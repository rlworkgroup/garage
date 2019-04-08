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
            n_epochs = 20
            n_epoch_cycles = 100
            sampler_batch_size = 10
            num_timesteps = n_epochs * n_epoch_cycles * sampler_batch_size
            env = TfEnv(gym.make("CartPole-v0"))
            replay_buffer = SimpleReplayBuffer(
                env_spec=env.spec,
                size_in_transitions=int(1e4),
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
                qf_lr=1e-4,
                discount=1.0,
                min_buffer_size=int(1e3),
                double_q=False,
                n_train_steps=10,
                n_epoch_cycles=n_epoch_cycles,
                target_network_update_freq=50,
                buffer_batch_size=32)

            runner.setup(algo, env)
            last_avg_ret = runner.train(
                n_epochs=n_epochs, n_epoch_cycles=n_epoch_cycles, batch_size=sampler_batch_size)
            assert last_avg_ret > 80

            env.close()
