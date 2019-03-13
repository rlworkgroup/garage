"""
An example to train a task with DQN algorithm.

Here it creates a gym environment CartPole, and trains a DQN with 50k steps.
"""
import gym
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import LocalRunner, run_experiment
from garage.exploration_strategies import EpsilonGreedyStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteMLPQFunction


def run_task(*_):
    """Run task."""
    # with LocalRunner() as runner:
    with tf.Session() as sess:
        max_path_length = 1
        num_timesteps = 20000

        env = TfEnv(normalize(gym.make("CartPole-v0")))

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=int(1e4),
            time_horizon=max_path_length)

        qf = DiscreteMLPQFunction(
            env_spec=env.spec, hidden_sizes=(64, 64), dueling=False)

        policy = DiscreteQfDerivedPolicy(env_spec=env.spec, qf=qf)

        epilson_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=env.spec,
            total_timesteps=num_timesteps,
            max_epsilon=1.0,
            min_epsilon=0.02,
            decay_ratio=0.1)

        dqn = DQN(
            env=env,
            policy=policy,
            qf=qf,
            exploration_strategy=epilson_greedy_strategy,
            replay_buffer=replay_buffer,
            max_path_length=max_path_length,
            num_timesteps=num_timesteps,
            qf_lr=1e-4,
            discount=1.0,
            min_buffer_size=int(1e3),
            double_q=False,
            target_network_update_freq=500,
            buffer_batch_size=32)

        # runner.setup(algo=dqn, env=env)        
        # runner.train()

        dqn.train(sess)


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
