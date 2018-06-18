import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
import gym
import numpy as np
import tensorflow as tf

from garage.misc.instrument import run_experiment_lite
from garage.tf.algos import DDPG
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.tf.policies import ContinuousMLPPolicy


def test_ddpg():
    env = gym.make('Pendulum-v0')

    def run_task(*_):
        action_noise = OUStrategy(env, sigma=0.2)

        actor_net = ContinuousMLPPolicy(
            env_spec=env,
            name="Actor",
            hidden_sizes=[64, 64],
            hidden_nonlinearity=tf.nn.relu,
        )

        critic_net = ContinuousMLPQFunction(
            env_spec=env,
            name="Critic",
            hidden_sizes=[64, 64],
            hidden_nonlinearity=tf.nn.relu,
        )

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
            optimizer=tf.train.AdamOptimizer)

        ddpg.train()

    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        plot=False,
    )

    nb_actions = env.action_space.shape[-1]

    stddev = 0.2
    action_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    memory = Memory(
        limit=int(1e6),
        action_shape=env.action_space.shape,
        observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=False)
    actor = Actor(nb_actions, layer_norm=False)

    training.train(
        env=env,
        param_noise=None,
        action_noise=action_noise,
        actor=actor,
        critic=critic,
        memory=memory,
        gamma=0.99,
        tau=0.001,
        normalize_returns=False,
        normalize_observations=True,
        nb_eval_steps=0,
        batch_size=64,
        nb_epochs=500,
        nb_epoch_cycles=20,
        nb_train_steps=50,
        nb_rollout_steps=100,
        critic_l2_reg=0.,
        actor_lr=1e-4,
        critic_lr=1e-3,
        clip_norm=None,
        reward_scale=1.,
        popart=False,
        render_eval=False,
        render=False)


test_ddpg()
