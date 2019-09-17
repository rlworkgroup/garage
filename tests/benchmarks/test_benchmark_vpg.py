"""
This script creates a regression test over
garage-PyTorch-VPG, garage-TF-VPG and baselines-VPG.
It get Mujoco1M benchmarks from baselines benchmark, and test each task in
its trial times on garage model and baselines model. For each task, there will
be `trial` times with different random seeds. For each trial, there will be two
log directories corresponding to baselines and garage. And there will be a plot
plotting the average return curve from baselines and garage.
"""

import datetime
import json
import os.path as osp
import random

from baselines.bench import benchmarks
import dowel
import dowel.logger as dowel_logger
import gym
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import tensorflow as tf
import torch

from garage.envs import normalize
from garage.experiment import deterministic, LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import VPG as TF_VPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy as TF_GMP
from garage.torch.algos import VPG as PyTorch_VPG
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from tests import helpers as Rh
from tests.fixtures import snapshot_config

hyper_parameters = {
    'hidden_sizes': [64, 64],
    'center_adv': True,
    'learning_rate': 1e-2,
    'discount': 0.99,
    'n_epochs': 250,
    'max_path_length': 500
}


def run_garage_pytorch(env, seed, log_dir):
    """Create garage PyTorch VPG model and training."""
    env = TfEnv(normalize(env))

    deterministic.set_seed(seed)

    runner = LocalRunner(snapshot_config)

    policy = PyTorch_GMP(
        env.spec,
        hidden_sizes=hyper_parameters['hidden_sizes'],
        hidden_nonlinearity=torch.tanh,
        # init_std=1e-6,
        output_nonlinearity=None)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = PyTorch_VPG(env_spec=env.spec,
                       policy=policy,
                       optimizer=torch.optim.Adam,
                       baseline=baseline,
                       max_path_length=hyper_parameters['max_path_length'],
                       discount=hyper_parameters['discount'],
                       center_adv=hyper_parameters['center_adv'],
                       policy_lr=hyper_parameters['learning_rate'])

    # Set up logger since we are not using run_experiment
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    dowel_logger.add_output(dowel.StdOutput())
    dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
    dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

    runner.setup(algo, env)
    runner.train(n_epochs=hyper_parameters['n_epochs'], batch_size=2048)

    dowel_logger.remove_all()

    return tabular_log_file


def run_garage_tf(env, seed, log_dir):
    """Create garage TensorFlow VPG model and training."""
    deterministic.set_seed(seed)

    with LocalTFRunner(snapshot_config) as runner:
        env = TfEnv(normalize(env))

        policy = TF_GMP(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['hidden_sizes'],
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TF_VPG(env_spec=env.spec,
                      policy=policy,
                      baseline=baseline,
                      max_path_length=hyper_parameters['max_path_length'],
                      discount=hyper_parameters['discount'],
                      center_adv=hyper_parameters['center_adv'],
                      optimizer_args=dict(
                          tf_optimizer_args=dict(
                              learning_rate=hyper_parameters['learning_rate']),
                          verbose=True))  # yapf: disable

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'], batch_size=2048)

        dowel_logger.remove_all()

        return tabular_log_file


def create_json(csvs, trails, seeds, xs, ys, factors, names):
    """Convert garage and benchmark csv outputs to json format."""
    task_result = {}
    for trail in range(trails):
        trail_seed = 'trail_%d' % (trail + 1)
        task_result['seed'] = seeds[trail]
        task_result[trail_seed] = {}

        dfs = (json.loads(pd.read_csv(csv[trail]).to_json()) for csv in csvs)
        task_result[trail_seed] = {
            name: {
                'time_steps': [float(val) * factor for val in df[x].values()],
                'return': df[y]
            }
            for df, x, y, factor, name in zip(dfs, xs, ys, factors, names)
        }
    return task_result


def plot(csvs,
         xs,
         ys,
         trials,
         seeds,
         plt_file,
         env_id,
         x_label,
         y_label,
         names,
         smooth=False,
         rolling=5):
    """Plot benchmark from csv files of garage and baselines."""
    assert all(len(x) == len(csvs[0]) for x in csvs)
    for trial in range(trials):
        seed = seeds[trial]

        for csv, x, y, name in zip(csvs, xs, ys, names):
            df = pd.read_csv(csv[trial])

            if smooth:
                rolling_y = df[y].rolling(rolling, min_periods=1)
                y_mean, y_deviation = rolling_y.mean(), rolling_y.std().fillna(
                    0)**2
                plt.plot(df[x],
                         y_mean,
                         label='garage_trial%d_seed%d%s' %
                         (trial + 1, seed, name))
                plt.fill_between(y_deviation.index,
                                 (y_mean - 2 * y_deviation)[0],
                                 (y_mean + 2 * y_deviation)[0],
                                 alpha=.1)

            else:
                plt.plot(df[x],
                         df[y],
                         label='garage_trial%d_seed%d%s' %
                         (trial + 1, seed, name))

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(env_id)

    plt.savefig(plt_file)
    plt.close()


@pytest.mark.huge
def test_benchmark_vpg():
    """Compare benchmarks between garage and baselines."""
    mujoco1m = benchmarks.get_benchmark('Mujoco1M')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    benchmark_dir = './data/local/benchmarks/vpg/%s/' % timestamp
    result_json = {}
    for task in mujoco1m['tasks']:
        env_id = task['env_id']

        env = gym.make(env_id)

        seeds = random.sample(range(100), task['trials'])

        task_dir = osp.join(benchmark_dir, env_id)
        plt_file = osp.join(benchmark_dir, '{}_benchmark.png'.format(env_id))

        garage_tf_csvs = []
        garage_pytorch_csvs = []

        for trial in range(task['trials']):
            seed = seeds[trial]

            trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
            garage_tf_dir = trial_dir + '/garage/tf'
            garage_pytorch_dir = trial_dir + '/garage/pytorch'

            env.reset()
            garage_pytorch_csv = run_garage_pytorch(env, seed,
                                                    garage_pytorch_dir)

            with tf.Graph().as_default():
                # Run garage algorithms
                env.reset()
                garage_tf_csv = run_garage_tf(env, seed, garage_tf_dir)

            garage_tf_csvs.append(garage_tf_csv)
            garage_pytorch_csvs.append(garage_pytorch_csv)

        env.close()

        plot([garage_tf_csvs, garage_pytorch_csvs], ['Iteration', 'Iteration'],
             ['AverageReturn', 'AverageReturn'],
             trials=task['trials'],
             seeds=seeds,
             plt_file=plt_file,
             env_id=env_id,
             x_label='Iteration',
             y_label='AverageReturn',
             names=['garage-tf', 'garage-pytorch'],
             smooth=True)

        result_json[env_id] = create_json(
            [garage_tf_csvs, garage_pytorch_csvs],
            seeds=seeds,
            trails=task['trials'],
            xs=['Iteration', 'Iteration'],
            ys=['AverageReturn', 'AverageReturn'],
            factors=[2048, 2047],
            names=['garage-tf', 'garage-pytorch'])

    Rh.write_file(result_json, 'VPG')
