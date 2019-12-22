"""This script creates a regression test over garage-TRPO and baselines-TRPO.

Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length. We also need to change the
garage.tf.samplers.BatchSampler to smooth the reward curve.
"""
import datetime
import multiprocessing
import os.path as osp
import random

from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common import set_global_seeds
from baselines.common.tf_util import _PLACEHOLDER_CACHE
from baselines.logger import configure
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import dowel
from dowel import logger as dowel_logger
import gym
import pytest
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
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from tests import benchmark_helper
from tests.fixtures import snapshot_config
import tests.helpers as Rh
from tests.wrappers import AutoStopEnv

hyper_parameters = {
    'hidden_sizes': [32, 32],
    'max_kl': 0.01,
    'gae_lambda': 0.97,
    'discount': 0.99,
    'max_path_length': 100,
    'n_epochs': 999,
    'batch_size': 1024,
    'n_trials': 5
}


class TestBenchmarkPPO:  # pylint: disable=too-few-public-methods
    """Compare benchmarks between garage and baselines."""

    @pytest.mark.huge
    def test_benchmark_trpo(self):  # pylint: disable=no-self-use
        """Compare benchmarks between garage and baselines."""
        mujoco1m = benchmarks.get_benchmark('Mujoco1M')

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/trpo/%s/' % timestamp
        result_json = {}
        for task in mujoco1m['tasks']:
            env_id = task['env_id']
            env = gym.make(env_id)
            baseline_env = AutoStopEnv(env_name=env_id, max_path_length=100)
            seeds = random.sample(range(100), hyper_parameters['n_trials'])
            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_id))
            baselines_csvs = []
            garage_tf_csvs = []
            garage_pytorch_csvs = []

            for trial in range(hyper_parameters['n_trials']):
                _PLACEHOLDER_CACHE.clear()
                seed = seeds[trial]
                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                garage_tf_dir = trial_dir + '/garage'
                garage_pytorch_dir = trial_dir + '/garage_pytorch'
                baselines_dir = trial_dir + '/baselines'

                # Run garage algorithms
                env.reset()
                garage_pytorch_csv = run_garage_pytorch(
                    env, seed, garage_pytorch_dir)

                # pylint: disable=not-context-manager
                with tf.Graph().as_default():
                    env.reset()
                    garage_tf_csv = run_garage(env, seed, garage_tf_dir)

                    # Run baseline algorithms
                    baseline_env.reset()
                    baselines_csv = run_baselines(baseline_env, seed,
                                                  baselines_dir)

                garage_tf_csvs.append(garage_tf_csv)
                garage_pytorch_csvs.append(garage_pytorch_csv)
                baselines_csvs.append(baselines_csv)

            env.close()

            benchmark_helper.plot_average_over_trials(
                [baselines_csvs, garage_tf_csvs, garage_pytorch_csvs],
                [
                    'eprewmean', 'Evaluation/AverageReturn',
                    'Evaluation/AverageReturn'
                ],
                plt_file=plt_file,
                env_id=env_id,
                x_label='Iteration',
                y_label='Evaluation/AverageReturn',
                names=['baseline', 'garage-TensorFlow', 'garage-PyTorch'],
            )

            result_json[env_id] = benchmark_helper.create_json(
                [baselines_csvs, garage_tf_csvs, garage_pytorch_csvs],
                seeds=seeds,
                trials=hyper_parameters['n_trials'],
                xs=['nupdates', 'Iteration', 'Iteration'],
                ys=[
                    'eprewmean', 'Evaluation/AverageReturn',
                    'Evaluation/AverageReturn'
                ],
                factors=[hyper_parameters['batch_size']] * 3,
                names=['baseline', 'garage-TF', 'garage-PT'])

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
                         hidden_sizes=hyper_parameters['hidden_sizes'],
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = PyTorch_TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_kl_step=hyper_parameters['max_kl'],
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


def run_garage(env, seed, log_dir):
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


def run_baselines(env, seed, log_dir):
    """Create Baseline model and training.

    Args:
        env (dict): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    ncpu = max(multiprocessing.cpu_count() // 2, 1)
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.compat.v1.Session(config=config).__enter__()

    # Set up logger for baselines
    configure(dir=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    baselines_logger.info('rank {}: seed={}, logdir={}'.format(
        0, seed, baselines_logger.get_dir()))

    set_global_seeds(seed)

    def policy_fn(name, ob_space, ac_space):
        """Create policy for baselines.

        Args:
            name (str): Policy name.
            ob_space (gym.spaces.Box) : Observation space.
            ac_space (gym.spaces.Box) : Action space.

        Returns:
            baselines.ppo1.mlp_policy: MLP policy for baselines.

        """
        return MlpPolicy(name=name,
                         ob_space=ob_space,
                         ac_space=ac_space,
                         hid_size=hyper_parameters['hidden_sizes'][0],
                         num_hid_layers=len(hyper_parameters['hidden_sizes']))

    trpo_mpi.learn(env,
                   policy_fn,
                   timesteps_per_batch=hyper_parameters['batch_size'],
                   max_kl=hyper_parameters['max_kl'],
                   cg_iters=10,
                   cg_damping=0.1,
                   max_timesteps=(hyper_parameters['batch_size'] *
                                  hyper_parameters['n_epochs']),
                   gamma=hyper_parameters['discount'],
                   lam=hyper_parameters['gae_lambda'],
                   vf_iters=5,
                   vf_stepsize=1e-3)

    return osp.join(log_dir, 'progress.csv')
