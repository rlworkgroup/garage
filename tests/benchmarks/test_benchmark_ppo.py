"""
This script creates a regression test over garage-PPO and baselines-PPO.

Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So you need to change the baselines source code to make it
stops at length 100.
"""
import datetime
import multiprocessing
import os.path as osp
import random
import unittest

from baselines import bench
from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.logger import configure
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import gym
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from garage.envs import normalize
from garage.misc import ext
from garage.misc import logger as garage_logger
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


class TestBenchmarkPPO(unittest.TestCase):
    def test_benchmark_ppo(self):
        """
        Compare benchmarks between garage and baselines.

        :return:
        """
        mujoco1m = benchmarks.get_benchmark("Mujoco1M")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        benchmark_dir = "./benchmark_ppo/%s/" % timestamp
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

                # Run garage algorithms
                garage_csv = run_garage(env, seed, garage_dir)

                # Run baselines algorithms
                env.reset()
                baselines_csv = run_baselines(env, seed, baselines_dir)

                garage_csvs.append(garage_csv)
                baselines_csvs.append(baselines_csv)

            plot(
                b_csvs=baselines_csvs,
                g_csvs=garage_csvs,
                g_x="Iteration",
                g_y="AverageReturn",
                b_x="nupdates",
                b_y="eprewmean",
                trails=task["trials"],
                seeds=seeds,
                plt_file=plt_file,
                env_id=env_id)

    test_benchmark_ppo.huge = True


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
        env = TfEnv(normalize(env))

        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(64, 64),
                use_trust_region=True,
            ),
        )

        algo = PPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=2048,
            max_path_length=100,
            n_itr=488,
            discount=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            policy_ent_coeff=0.0,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                tf_optimizer_args=dict(
                    learning_rate=3e-4,
                    epsilon=1e-5,
                ),
            ),
            plot=False,
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, "progress.csv")
        garage_logger.add_tabular_output(tabular_log_file)
        garage_logger.set_tensorboard_dir(log_dir)

        algo.train()

        garage_logger.remove_tabular_output(tabular_log_file)

        return tabular_log_file


def run_baselines(env, seed, log_dir):
    """
    Create baselines model and training.

    Replace the ppo and its training with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return
    """
    ncpu = max(multiprocessing.cpu_count() // 2, 1)
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=ncpu,
        inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    # Set up logger for baselines
    configure(dir=log_dir)
    baselines_logger.info('rank {}: seed={}, logdir={}'.format(
        0, seed, baselines_logger.get_dir()))

    def make_env():
        monitor = bench.Monitor(
            env, baselines_logger.get_dir(), allow_early_resets=True)
        return monitor

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(
        policy=policy,
        env=env,
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        vf_coef=0.5,
        max_grad_norm=0.5,
        cliprange=0.1,
        total_timesteps=int(1e6))

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
