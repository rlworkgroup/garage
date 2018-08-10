"""This script creates a regression test over garage-HER and baselines-HER."""
import datetime
import os.path as osp
import random
import unittest

from baselines.bench import benchmarks
from baselines.her.experiment.config import CACHED_ENVS
from baselines.her.experiment.config import DEFAULT_PARAMS as BASELINES_PARAMS
from baselines.her.experiment.train import launch
import gym
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from garage.misc import ext
from garage.misc import logger as garage_logger
from garage.tf.algos import DDPG
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction

# Hyperparams for baselines and garage
params = {
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "actor_hidden_sizes": [256, 256, 256],
    "critic_hidden_sizes": [256, 256, 256],
    "n_epochs": 50,
    "n_epoch_cycles": 20,
    "n_rollout_steps": 100,
    "n_train_steps": 40,
    "discount": 0.9,
    "tau": 0.05,
    "replay_buffer_size": int(1e6),
    "sigma": 0.2,
}

BASELINES_PARAMS["rollout_batch_size"] = 1


class TestBenchmarkHER(unittest.TestCase):
    def test_benchmark_her(self):
        """
        Compare benchmarks between garage and baselines.

        :return:
        """
        mujoco1m = benchmarks.get_benchmark("HerDdpg")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        benchmark_dir = "./benchmark_her/%s/" % timestamp
        for task in mujoco1m["tasks"]:
            env_id = task["env_id"]
            env = gym.make(env_id)
            seeds = random.sample(range(100), task["trials"])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                "{}_benchmark.png".format(env_id))
            baselines_csvs = []
            garage_csvs = []

            for trail in range(task["trials"]):
                env.reset()
                seed = seeds[trail]

                trail_dir = task_dir + "/trail_%d_seed_%d" % (trail + 1, seed)
                garage_dir = trail_dir + "/garage"
                baselines_dir = trail_dir + "/baselines"

                CACHED_ENVS.clear()
                baselines_csv = run_baselines(env_id, seed, baselines_dir)

                garage_csv = run_garage(env, seed, garage_dir)
                env.close()

                garage_csvs.append(garage_csv)
                baselines_csvs.append(baselines_csv)

            plot(
                b_csvs=baselines_csvs,
                g_csvs=garage_csvs,
                g_x="Epoch",
                g_y="AverageSuccessRate",
                b_x="epoch",
                b_y="train/success_rate",
                trails=task["trials"],
                seeds=seeds,
                plt_file=plt_file,
                env_id=env_id)

    test_benchmark_her.huge = True


def run_garage(env, seed, log_dir):
    """
    Create garage model and training.

    Replace the ppo with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return:
    """
    ext.set_seed(seed)

    with tf.Graph().as_default():
        # env = TfEnv(normalize(env))

        action_noise = OUStrategy(env, sigma=params["sigma"])

        actor_net = ContinuousMLPPolicy(
            env_spec=env,
            name="Actor",
            hidden_sizes=params["actor_hidden_sizes"],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            input_include_goal=True,
        )

        critic_net = ContinuousMLPQFunction(
            env_spec=env,
            name="Critic",
            hidden_sizes=params["critic_hidden_sizes"],
            hidden_nonlinearity=tf.nn.relu,
            input_include_goal=True,
        )

        algo = DDPG(
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
            critic_optimizer=tf.train.AdamOptimizer,
            use_her=True,
            batch_size=256,
            clip_obs=200.,
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, "progress.csv")
        garage_logger.add_tabular_output(tabular_log_file)
        garage_logger.set_tensorboard_dir(log_dir)

        algo.train()

        garage_logger.remove_tabular_output(tabular_log_file)

        return tabular_log_file


def run_baselines(env_id, seed, log_dir):
    """
    Create baselines model and training.

    Replace the ppo and its training with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return
    """
    launch_params = {
        "env": env_id,
        "logdir": log_dir,
        "n_epochs": params["n_epochs"],
        "num_cpu":
        1,  # For FetchReachEnv, the performance is not relevant to num_cpu
        "seed": seed,
        "policy_save_interval": 0,
        "replay_strategy": "future",
        "clip_return": 1,
    }
    launch(**launch_params)

    return osp.join(log_dir, "progress.csv")


def plot(b_csvs, g_csvs, g_x, g_y, b_x, b_y, trails, seeds, plt_file, env_id):
    """
    Plot benchmark from csv files of garage and baselines.

    :param b_csvs: A list contains all csv files in the task.
    :param g_csvs: A list contains all csv files in the task.
    :param g_x: X column names of garage csv.
    :param g_y: Y column names of garage csv.
    :param b_x: X column names of baselines csv.
    :param b_y: Y column names of baselines csv.
    :param trails: Number of trails in the task.
    :param seeds: A list contains all the seeds in the task.
    :param plt_file: Path of the plot png file.
    :param env_id: String contains the id of the environment.
    :return:
    """
    assert len(b_csvs) == len(g_csvs)
    for trail in range(trails):
        seed = seeds[trail]

        df_g = pd.read_csv(g_csvs[trail])
        df_b = pd.read_csv(b_csvs[trail])

        plt.plot(
            df_g[g_x],
            df_g[g_y],
            label="garage_trail%d_seed%d" % (trail + 1, seed))
        plt.plot(
            df_b[b_x],
            df_b[b_y],
            label="baselines_trail%d_seed%d" % (trail + 1, seed))

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("AverageReturn")
    plt.title(env_id)

    plt.savefig(plt_file)
    plt.close()
