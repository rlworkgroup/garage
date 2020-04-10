"""This script creates a regression test over garage-TRPO and baselines-TRPO.

Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length. We also need to change the
garage.tf.samplers.BatchSampler to smooth the reward curve.
"""
import datetime
import os.path as osp
import random

from baselines.bench import benchmarks
from baselines.common.tf_util import _PLACEHOLDER_CACHE
import dowel
from dowel import logger as dowel_logger
import gym
import tensorflow as tf
import torch

from garage.envs import normalize
from garage.experiment import deterministic, LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
from garage.torch.algos import TRPO as PyTorch_TRPO
from garage.torch.optimizers import (ConjugateGradientOptimizer,
                                     OptimizerWrapper)
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from garage.torch.value_functions import GaussianMLPValueFunction
from tests import benchmark_helper
from tests.fixtures import snapshot_config
import tests.helpers as Rh

hyper_parameters = {
    'hidden_sizes': [32, 32],
    'max_kl': 0.01,
    'gae_lambda': 0.97,
    'discount': 0.99,
    'max_path_length': 100,
    'n_epochs': 500,
    'batch_size': 1024,
    'n_trials': 4
}


class BenchmarkTRPO:  # pylint: disable=too-few-public-methods
    """Compare benchmarks between garage and baselines."""

    def benchmark_trpo(self):  # pylint: disable=no-self-use
        """Compare benchmarks between garage and baselines."""
        mujoco1m = benchmarks.get_benchmark('Mujoco1M')

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/trpo/%s/' % timestamp
        result_json = {}
        for task in mujoco1m['tasks']:
            env_id = task['env_id']
            env = gym.make(env_id)
            seeds = random.sample(range(100), hyper_parameters['n_trials'])
            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_id))
            garage_tf_csvs = []
            garage_pytorch_csvs = []

            for trial in range(hyper_parameters['n_trials']):
                _PLACEHOLDER_CACHE.clear()
                seed = seeds[trial]
                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                garage_tf_dir = trial_dir + '/garage_tf'
                garage_pytorch_dir = trial_dir + '/garage_pytorch'

                # Run garage algorithms
                env.reset()
                garage_pytorch_csv = run_garage_pytorch(
                    env, seed, garage_pytorch_dir)

                # pylint: disable=not-context-manager
                with tf.Graph().as_default():
                    env.reset()
                    garage_tf_csv = run_garage_tf(env, seed, garage_tf_dir)

                garage_tf_csvs.append(garage_tf_csv)
                garage_pytorch_csvs.append(garage_pytorch_csv)

            env.close()

            benchmark_helper.plot_average_over_trials(
                [garage_tf_csvs, garage_pytorch_csvs],
                ['Evaluation/AverageReturn'] * 2,
                plt_file=plt_file,
                env_id=env_id,
                x_label='Iteration',
                y_label='AverageReturn',
                names=['garage-TensorFlow', 'garage-PyTorch'],
            )

            result_json[env_id] = benchmark_helper.create_json(
                [garage_tf_csvs, garage_pytorch_csvs],
                seeds=seeds,
                trials=hyper_parameters['n_trials'],
                xs=['Evaluation/Iteration'] * 2,
                ys=['Evaluation/AverageReturn'] * 2,
                factors=[hyper_parameters['batch_size']] * 2,
                names=['garage-TF', 'garage-PT'])

        Rh.write_file(result_json, 'TRPO')


def run_garage_pytorch(env, seed, log_dir):
    """Create garage PyTorch PPO model and training.

    Args:
        env (dict): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    env = TfEnv(normalize(env))

    deterministic.set_seed(seed)

    runner = LocalRunner(snapshot_config)

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=(32, 32),
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    policy_optimizer = OptimizerWrapper(
        (ConjugateGradientOptimizer, dict(max_constraint_value=0.01)), policy)
    vf_optimizer = OptimizerWrapper((torch.optim.Adam, dict(lr=2.5e-4)),
                                    value_function,
                                    max_optimization_epochs=10,
                                    minibatch_size=64)

    algo = PyTorch_TRPO(env_spec=env.spec,
                        policy=policy,
                        value_function=value_function,
                        policy_optimizer=policy_optimizer,
                        vf_optimizer=vf_optimizer,
                        max_path_length=hyper_parameters['max_path_length'],
                        discount=hyper_parameters['discount'],
                        gae_lambda=hyper_parameters['gae_lambda'])

    # Set up logger since we are not using run_experiment
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    dowel_logger.add_output(dowel.StdOutput())
    dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
    dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

    runner.setup(algo, env)
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=hyper_parameters['batch_size'])

    dowel_logger.remove_all()

    return tabular_log_file


def run_garage_tf(env, seed, log_dir):
    """Create garage Tensorflow PPO model and training.

    Args:
        env (dict): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    deterministic.set_seed(seed)

    with LocalTFRunner(snapshot_config) as runner:
        env = TfEnv(normalize(env))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['hidden_sizes'],
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=hyper_parameters['max_path_length'],
                    discount=hyper_parameters['discount'],
                    gae_lambda=hyper_parameters['gae_lambda'],
                    max_kl_step=hyper_parameters['max_kl'])

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['batch_size'])

        dowel_logger.remove_all()

        return tabular_log_file
