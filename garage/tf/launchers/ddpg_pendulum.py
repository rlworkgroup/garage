import gym
import tensorflow as tf

from garage.misc.instrument import run_experiment
from garage.tf.algos import DDPG
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction


def run_task(*_):

    env = gym.make('Pendulum-v0')

    action_noise = OUStrategy(env)

    actor_net = ContinuousMLPPolicy(
        env_spec=env,
        name="Actor",
        hidden_sizes=[64, 64],
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh)

    critic_net = ContinuousMLPQFunction(
        env_spec=env,
        name="Critic",
        hidden_sizes=[64, 64],
        hidden_nonlinearity=tf.nn.relu)

    ddpg = DDPG(
        env,
        actor=actor_net,
        critic=critic_net,
        plot=False,
        n_epochs=500,
        n_epoch_cycles=20,
        n_rollout_steps=100,
        n_train_steps=50,
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
