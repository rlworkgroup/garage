"""
This is an example to train a task with DDPG + HER algorithm.

Here it creates a gym environment FetchReach.

Results (may vary by seed):
    AverageSuccessRate: 0.9
    RiseTime: epoch 8
"""
import gym
import tensorflow as tf

from garage.misc.instrument import run_experiment
from garage.tf.algos import DDPG
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction


def run_task(*_):
    """
    Wrap DDPG training task in the run_task function.

    :param _:
    :return:
    """
    env = gym.make('FetchReach-v1')

    action_noise = OUStrategy(env, sigma=0.2)

    actor_net = ContinuousMLPPolicy(
        env_spec=env,
        name="Actor",
        hidden_sizes=[256, 256, 256],
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
        input_include_goal=True,
    )

    critic_net = ContinuousMLPQFunction(
        env_spec=env,
        name="Critic",
        hidden_sizes=[256, 256, 256],
        hidden_nonlinearity=tf.nn.relu,
        input_include_goal=True,
    )

    ddpg = DDPG(
        env,
        actor=actor_net,
        actor_lr=1e-3,
        critic_lr=1e-3,
        critic=critic_net,
        plot=False,
        target_update_tau=0.05,
        n_epochs=50,
        n_epoch_cycles=20,
        n_rollout_steps=100,
        n_train_steps=40,
        discount=0.9,
        replay_buffer_size=int(1e6),
        min_buffer_size=int(1e4),
        exploration_strategy=action_noise,
        actor_optimizer=tf.train.AdamOptimizer,
        critic_optimizer=tf.train.AdamOptimizer,
        use_her=True,
        batch_size=256,
        clip_obs=200.,
    )

    ddpg.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
