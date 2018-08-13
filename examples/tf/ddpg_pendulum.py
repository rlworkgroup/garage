"""
This is an example to train a task with DDPG algorithm.

Here it creates a gym environment InvertedPendulum. And uses a DDPG with 1M
steps.

Results:
    AverageReturn: 250
    RiseTime: epoch 300
"""
import gym
import tensorflow as tf

from garage.misc.instrument import run_experiment
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction


def run_task(*_):
    """
    Wrap DDPG training task in the run_task function.

    :param _:
    :return:
    """
    env = TfEnv(gym.make('InvertedDoublePendulum-v2'))

    action_noise = OUStrategy(env, sigma=0.2)

    actor_net = ContinuousMLPPolicy(
        env_spec=env.spec,
        name="Actor",
        hidden_sizes=[64, 64],
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh)

    critic_net = ContinuousMLPQFunction(
        env_spec=env.spec,
        name="Critic",
        hidden_sizes=[64, 64],
        hidden_nonlinearity=tf.nn.relu)

    ddpg = DDPG(
        env,
        actor=actor_net,
        actor_lr=1e-4,
        critic_lr=1e-3,
        critic=critic_net,
        plot=False,
        target_update_tau=1e-2,
        n_epochs=500,
        n_epoch_cycles=20,
        n_rollout_steps=100,
        n_train_steps=50,
        discount=0.9,
        replay_buffer_size=int(1e6),
        min_buffer_size=int(1e4),
        exploration_strategy=action_noise,
        actor_optimizer=tf.train.AdamOptimizer,
        critic_optimizer=tf.train.AdamOptimizer)

    ddpg.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
