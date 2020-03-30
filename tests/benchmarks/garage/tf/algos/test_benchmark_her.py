"""This script creates a regression test over garage-HER and baselines-HER."""

import datetime  # noqa: I100
import os
import os.path as osp
import random

import pytest
try:
    # pylint: disable=unused-import
    import mujoco_py  # noqa: F401
except ImportError:
    pytest.skip('To use mujoco-based features, please install garage[mujoco].',
                allow_module_level=True)
except Exception:  # pylint: disable=broad-except
    pytest.skip(
        'Skipping tests, failed to import mujoco. Do you have a '
        'valid mujoco key installed?',
        allow_module_level=True)

try:
    from baselines.bench import benchmarks  # noqa: I100, I202
    from baselines.her.experiment.config import CACHED_ENVS
    from baselines.her.experiment.config import DEFAULT_PARAMS \
        as BASELINES_PARAMS
    from baselines.her.her import learn
except ImportError:
    pytest.skip('To run test_benchmark_her install mpi4py',
                allow_module_level=True)

import dowel  # noqa: I100, I202
from dowel import logger
import gym
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import deterministic
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import HerReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from tests.fixtures import snapshot_config

# Hyperparams for baselines and garage
params = {
    'policy_lr': 1e-3,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [256, 256, 256],
    'qf_hidden_sizes': [256, 256, 256],
    'n_epochs': 50,
    'steps_per_epoch': 20,
    'n_rollout_steps': 100,
    'n_train_steps': 40,
    'discount': 0.9,
    'tau': 0.05,
    'replay_buffer_size': int(1e6),
    'sigma': 0.2,
}

BASELINES_PARAMS['rollout_batch_size'] = 1


@pytest.mark.huge
def test_benchmark_her():
    """Compare benchmarks between garage and baselines."""

    mujoco1m = benchmarks.get_benchmark('Fetch1M')
    print(mujoco1m)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    benchmark_dir = osp.join(os.getcwd(), 'data', 'local', 'benchmarks', 'her',
                             timestamp)
    for task in mujoco1m['tasks']:
        env_id = task['env_id']
        env = gym.make(env_id)
        seeds = random.sample(range(100), task['trials'])

        task_dir = osp.join(benchmark_dir, env_id)
        plt_file = osp.join(benchmark_dir, '{}_benchmark.png'.format(env_id))
        baselines_csvs = []
        garage_csvs = []

        for trial in range(task['trials']):
            seed = seeds[trial]

            trial_dir = osp.join(task_dir,
                                 'trial_{}_seed_{}'.format(trial + 1, seed))
            garage_dir = osp.join(trial_dir, 'garage')
            baselines_dir = osp.join(trial_dir, 'baselines')

            with tf.Graph().as_default():
                garage_csv = run_garage(env, seed, garage_dir)

                CACHED_ENVS.clear()
                baselines_csv = run_baselines(env, seed, baselines_dir)

            garage_csvs.append(garage_csv)
            baselines_csvs.append(baselines_csv)

        env.close()

        plot(b_csvs=baselines_csvs,
             g_csvs=garage_csvs,
             g_x='Epoch',
             g_y='AverageSuccessRate',
             b_x='epoch',
             b_y='train/success_rate',
             trials=task['trials'],
             seeds=seeds,
             plt_file=plt_file,
             env_id=env_id)


def run_garage(env, seed, log_dir):
    """Create garage model and training.

    Replace the ppo with the algorithm you want to run.

    Args:
        env (gym.Env): Environment of the task.
        seed (int): Random seed for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: The log file path.

    """
    deterministic.set_seed(seed)
    env.reset()

    with LocalTFRunner(snapshot_config) as runner:
        env = TfEnv(normalize(env))

        action_noise = OUStrategy(env.spec, sigma=params['sigma'])

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=params['policy_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
        )

        qf = ContinuousMLPQFunction(
            env_spec=env.spec,
            hidden_sizes=params['qf_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
        )

        replay_buffer = HerReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=params['replay_buffer_size'],
            time_horizon=params['n_rollout_steps'],
            replay_k=0.4,
            reward_fun=env.compute_reward,
        )

        algo = DDPG(
            env_spec=env.spec,
            policy=policy,
            qf=qf,
            replay_buffer=replay_buffer,
            steps_per_epoch=params['steps_per_epoch'],
            policy_lr=params['policy_lr'],
            qf_lr=params['qf_lr'],
            target_update_tau=params['tau'],
            n_train_steps=params['n_train_steps'],
            discount=params['discount'],
            exploration_strategy=action_noise,
            policy_optimizer=tf.train.AdamOptimizer,
            qf_optimizer=tf.train.AdamOptimizer,
            buffer_batch_size=256,
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        logger.add_output(dowel.StdOutput())
        logger.add_output(dowel.CsvOutput(tabular_log_file))
        logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo, env)
        runner.train(n_epochs=params['n_epochs'],
                     batch_size=params['n_rollout_steps'])

        logger.remove_all()

        return tabular_log_file


def run_baselines(env, seed, log_dir):
    """Create baselines model and training.

    Replace the ppo and its training with the algorithm you want to run.

    Args:
        env (gym.Env): Environment of the task.
        seed (int): Random seed for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: The log file path.

    """
    learn_params = {
        'network': 'mlp',
        'env': env,
        'total_timesteps': 1000,
        'logdir': log_dir,
        'num_cpu':
        1,  # For FetchReachEnv, the performance is not relevant to num_cpu
        'seed': seed,
        'policy_save_interval': 0,
        'replay_strategy': 'future',
        'clip_return': 1,
    }
    learn(**learn_params)

    return osp.join(log_dir, 'progress.csv')


def plot(b_csvs, g_csvs, g_x, g_y, b_x, b_y, trials, seeds, plt_file, env_id):
    """Plot benchmark from csv files of garage and baselines.

    Args:
        b_csvs (list[str]): A list contains all csv files in the task.
        g_csvs (list[str]): A list contains all csv files in the task.
        g_x (list[str]): X column names of garage csv.
        g_y (list[str]): Y column names of garage csv.
        b_x (list[str]): X column names of baselines csv.
        b_y (list[str]): Y column names of baselines csv.
        trials (int): Number of trials in the task.
        seeds (int): A list contains all the seeds in the task.
        plt_file (str): Path of the plot png file.
        env_id (str): String contains the id of the environment.

    """
    assert len(b_csvs) == len(g_csvs)
    for trial in range(trials):
        seed = seeds[trial]

        df_g = pd.read_csv(g_csvs[trial])
        df_b = pd.read_csv(b_csvs[trial])

        plt.plot(df_g[g_x],
                 df_g[g_y],
                 label='garage_trial{}_seed{}'.format(trial + 1, seed))
        plt.plot(df_b[b_x],
                 df_b[b_y],
                 label='baselines_trial{}_seed{}'.format(trial + 1, seed))

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Evaluation/AverageReturn')
    plt.title(env_id)

    plt.savefig(plt_file)
    plt.close()
