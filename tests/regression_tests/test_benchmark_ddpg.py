"""
This script creates a regression test over garage-DDPG and baselines-DDPG.

It get Mujoco1M benchmarks from baselines benchmark, and test each task in
its trail times on garage model and baselines model. For each task, there will
be `trail` times with different random seeds. For each trail, there will be two
log directories corresponding to baselines and garage. And there will be a plot
plotting the average return curve from baselines and garage.
"""
import datetime
import os.path as osp
import random

from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.ddpg.memory import Memory
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
import baselines.ddpg.training as training
from baselines.logger import configure
import gym
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import pandas as pd
import tensorflow as tf

from garage.misc import logger as garage_logger
from garage.tf.algos import DDPG
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction

# Hyperparams for baselines and garage
params = {
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "actor_hidden_sizes": [64, 64],
    "critic_hidden_sizes": [64, 64],
    "n_epochs": 500,
    "n_epoch_cycles": 20,
    "n_rollout_steps": 100,
    "n_train_steps": 50,
    "discount": 0.99,
    "tau": 1e-2,
    "replay_buffer_size": int(1e6),
    "sigma": 0.2,
}


def test_benchmark():
    """
    Compare benchmarks between garage and baselines.

    :return:
    """
    # Load Mujoco1M tasks, you can check other benchmarks here
    # https://github.com/openai/baselines/blob/master/baselines/bench/benchmarks.py
    mujoco1m = benchmarks.get_benchmark("Mujoco1M")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    for task in mujoco1m["tasks"]:
        env_id = task["env_id"]
        env = gym.make(env_id)
        seeds = random.sample(range(100), task["trials"])
        for i in range(task["trials"]):
            env.reset()
            seed = seeds[i]

            log_dir = "./benchmark/%s/" % timestamp
            trail_dir = log_dir + "%s_trail_%d_seed_%d" % (env_id, i + 1, seed)
            garage_dir = trail_dir + "/garage"
            baselines_dir = trail_dir + "/baselines"
            plt_file = trail_dir + "/benchmark.png"

            # Run garage algorithms
            csv_g = run_garage(env, seed, garage_dir)

            # Run baselines algorithms
            csv_b = run_baselines(env, seed, baselines_dir)

            plot(
                csv_g=csv_g,
                csv_b=csv_b,
                x_g="Epoch",
                y_g="AverageReturn",
                x_b="total/epochs",
                y_b="rollout/return",
                plt_file=plt_file)


def run_garage(env, seed, log_dir):
    """
    Create garage model and training.

    Replace the ddpg with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return:
    """
    env.seed(seed)

    with tf.Graph().as_default():
        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, "progress.csv")
        tensorboard_log_dir = osp.join(log_dir, "progress")
        garage_logger.add_tabular_output(tabular_log_file)
        garage_logger.set_tensorboard_dir(tensorboard_log_dir)

        # Set up params for ddpg
        action_noise = OUStrategy(env, sigma=params["sigma"])

        actor_net = ContinuousMLPPolicy(
            env_spec=env,
            name="Actor",
            hidden_sizes=params["actor_hidden_sizes"],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        critic_net = ContinuousMLPQFunction(
            env_spec=env,
            name="Critic",
            hidden_sizes=params["critic_hidden_sizes"],
            hidden_nonlinearity=tf.nn.relu)

        ddpg = DDPG(
            env,
            actor=actor_net,
            critic=critic_net,
            actor_lr=params["actor_lr"],
            critic_lr=params["critic_lr"],
            plot=False,
            target_update_tau=params["tau"],
            n_epochs=params["n_epochs"],
            n_epoch_cycles=params["n_epoch_cycles"],
            n_rollout_steps=params["n_rollout_steps"],
            n_train_steps=params["n_train_steps"],
            discount=params["discount"],
            replay_buffer_size=params["replay_buffer_size"],
            min_buffer_size=int(1e4),
            exploration_strategy=action_noise,
            actor_optimizer=tf.train.AdamOptimizer,
            critic_optimizer=tf.train.AdamOptimizer)

        ddpg.train()

        return tabular_log_file


def run_baselines(env, seed, log_dir):
    """
    Create baselines model and training.

    Replace the ddpg and its training with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return
    """
    rank = MPI.COMM_WORLD.Get_rank()
    env.seed(seed)

    # Set up logger for baselines
    configure(dir=log_dir)
    baselines_logger.info('rank {}: seed={}, logdir={}'.format(
        rank, seed, baselines_logger.get_dir()))

    # Set up params for baselines ddpg
    nb_actions = env.action_space.shape[-1]
    layer_norm = False

    action_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(nb_actions),
        sigma=float(params["sigma"]) * np.ones(nb_actions))
    memory = Memory(
        limit=params["replay_buffer_size"],
        action_shape=env.action_space.shape,
        observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    training.train(
        env=env,
        eval_env=None,
        param_noise=None,
        action_noise=action_noise,
        actor=actor,
        critic=critic,
        memory=memory,
        nb_epochs=params["n_epochs"],
        nb_epoch_cycles=params["n_epoch_cycles"],
        render_eval=False,
        reward_scale=1.,
        render=False,
        normalize_returns=False,
        normalize_observations=False,
        critic_l2_reg=0,
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"],
        popart=False,
        gamma=params["discount"],
        clip_norm=None,
        nb_train_steps=params["n_train_steps"],
        nb_rollout_steps=params["n_rollout_steps"],
        nb_eval_steps=100,
        batch_size=64)

    return osp.join(log_dir, "progress.csv")


def plot(csv_g, csv_b, x_g, y_g, x_b, y_b, plt_file):
    """
    Plot benchmark from csv files of garage and baselines.

    :param csv_g: Csv filename of garage.
    :param csv_b: Csv filename of baselines.
    :param x_g: X column names of garage csv.
    :param y_g: Y column names of garage csv.
    :param x_b: X column names of baselines csv.
    :param y_b: Y column names of baselines csv.
    :param plt_file: Path of the plot png file.
    :return:
    """
    df_g = pd.read_csv(csv_g)
    df_b = pd.read_csv(csv_b)

    plt.plot(df_g[x_g], df_g[y_g], label="garage")
    plt.plot(df_b[x_b], df_b[y_b], label="baselines")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("AverageReturn")

    plt.savefig(plt_file)
    plt.close()


test_benchmark()
