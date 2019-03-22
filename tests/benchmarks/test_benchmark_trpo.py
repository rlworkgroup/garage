"""
This script creates a regression test over garage-TRPO and baselines-TRPO.

Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length. We also need to change the
garage.tf.samplers.BatchSampler to smooth the reward curve.
"""
import datetime
import os.path as osp
import random
import unittest

from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common.tf_util import _PLACEHOLDER_CACHE
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import gym
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import deterministic
from garage.experiment import LocalRunner
from garage.misc import logger as garage_logger
from garage.tf.algos import TRPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.wrappers import AutoStopEnv


class TestBenchmarkPPO(unittest.TestCase):
    def test_benchmark_trpo(self):
        """
        Compare benchmarks between garage and baselines.

        :return:
        """

        mujoco1m = benchmarks.get_benchmark("Mujoco1M")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        benchmark_dir = "./data/local/benchmarks/trpo/%s/" % timestamp
        for task in mujoco1m["tasks"]:
            env_id = task["env_id"]
            env = gym.make(env_id)
            baseline_env = AutoStopEnv(env_name=env_id, max_path_length=100)

            seeds = random.sample(range(100), task["trials"])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                "{}_benchmark.png".format(env_id))
            baselines_csvs = []
            garage_csvs = []

            for trial in range(task["trials"]):
                _PLACEHOLDER_CACHE.clear()
                seed = seeds[trial]

                trial_dir = task_dir + "/trial_%d_seed_%d" % (trial + 1, seed)
                garage_dir = trial_dir + "/garage"
                baselines_dir = trial_dir + "/baselines"

                with tf.Graph().as_default():
                    # Run garage algorithms
                    env.reset()
                    garage_csv = run_garage(env, seed, garage_dir)

                    # Run baseline algorithms
                    baseline_env.reset()
                    baselines_csv = run_baselines(baseline_env, seed,
                                                  baselines_dir)

                garage_csvs.append(garage_csv)
                baselines_csvs.append(baselines_csv)

            plot(
                b_csvs=baselines_csvs,
                g_csvs=garage_csvs,
                g_x="Iteration",
                g_y="EpisodeRewardMean",
                b_x="EpThisIter",
                b_y="EpRewMean",
                trials=task["trials"],
                seeds=seeds,
                plt_file=plt_file,
                env_id=env_id)

            env.close()

    test_benchmark_trpo.huge = True


def run_garage(env, seed, log_dir):
    """
    Create garage model and training.

    Replace the trpo with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return:import baselines.common.tf_util as U
    """
    deterministic.set_seed(seed)

    with LocalRunner() as runner:
        env = TfEnv(normalize(env))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(32, 32),
                use_trust_region=True,
            ),
        )

        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.98,
            max_kl_step=0.01,
            policy_ent_coeff=0.0,
            plot=False,
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, "progress.csv")
        garage_logger.add_tabular_output(tabular_log_file)
        garage_logger.set_tensorboard_dir(log_dir)

        runner.setup(algo, env)
        runner.train(n_epochs=976, batch_size=1024)

        garage_logger.remove_tabular_output(tabular_log_file)

        return tabular_log_file


def run_baselines(env, seed, log_dir):
    """
    Create baselines model and training.

    Replace the trpo and its training with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return
    """

    with tf.Session().as_default():
        baselines_logger.configure(log_dir)

        def policy_fn(name, ob_space, ac_space):
            return MlpPolicy(
                name=name,
                ob_space=ob_space,
                ac_space=ac_space,
                hid_size=32,
                num_hid_layers=2)

        trpo_mpi.learn(
            env,
            policy_fn,
            timesteps_per_batch=1024,
            max_kl=0.01,
            cg_iters=10,
            cg_damping=0.1,
            max_timesteps=int(1e6),
            gamma=0.99,
            lam=0.98,
            vf_iters=5,
            vf_stepsize=1e-3)
        env.close()

    return osp.join(log_dir, "progress.csv")


def plot(b_csvs, g_csvs, g_x, g_y, b_x, b_y, trials, seeds, plt_file, env_id):
    """
    Plot benchmark from csv files of garage and baselines.

    :param b_csvs: A list contains all csv files in the task.
    :param g_csvs: A list contains all csv files in the task.
    :param g_x: X column names of garage csv.
    :param g_y: Y column names of garage csv.
    :param b_x: X column names of baselines csv.
    :param b_y: Y column names of baselines csv.
    :param trials: Number of trials in the task.
    :param seeds: A list contains all the seeds in the task.
    :param plt_file: Path of the plot png file.
    :param env_id: String contains the id of the environment.
    :return:
    """
    assert len(b_csvs) == len(g_csvs)
    for trial in range(trials):
        seed = seeds[trial]

        df_g = pd.read_csv(g_csvs[trial])
        df_b = pd.read_csv(b_csvs[trial])

        plt.plot(
            df_g[g_x],
            df_g[g_y],
            label="garage_trial%d_seed%d" % (trial + 1, seed))
        plt.plot(
            df_b[b_x],
            df_b[b_y],
            label="baselines_trial%d_seed%d" % (trial + 1, seed))

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("AverageReturn")
    plt.title(env_id)

    plt.savefig(plt_file)
    plt.close()
