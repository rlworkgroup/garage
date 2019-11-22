"""A regression test over PPO Algorithms."""

import datetime
import multiprocessing
import os.path as osp
import random

from baselines import bench
from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.logger import configure
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import dowel
from dowel import logger as dowel_logger
import gym
import pytest
import tensorflow as tf
import torch

from garage.envs import normalize
from garage.experiment import deterministic, LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO as TF_PPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy as TF_GMP
from garage.torch.algos import PPO as PyTorch_PPO
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from tests import benchmark_helper
from tests import helpers as Rh
from tests.fixtures import snapshot_config
from tests.wrappers import AutoStopEnv

hyper_parameters = {
    'hidden_sizes': [64, 64],
    'center_adv': True,
    'learning_rate': 3e-4,
    'lr_clip_range': 0.2,
    'gae_lambda': 0.97,
    'discount': 0.99,
    'n_epochs': 400,
    'max_path_length': 100,
    'batch_size': 2048,
    'n_trials': 10
}


class TestBenchmarkPPO:
    """A regression test over PPO Algorithms.
    (garage-PyTorch-PPO, garage-TensorFlow-PPO, and baselines-PPO2)

    It get Mujoco1M benchmarks from baselines benchmark, and test each task in
    its trial times on garage model and baselines model. For each task,
    there will
    be `trial` times with different random seeds. For each trial, there will
    be two
    log directories corresponding to baselines and garage. And there will be
    a plot
    plotting the average return curve from baselines and garage.
    """
    # pylint: disable=too-few-public-methods

    @pytest.mark.huge
    def test_benchmark_ppo(self):
        """Compare benchmarks between garage and baselines.

        Returns:

        """
        # pylint: disable=no-self-use
        mujoco1m = benchmarks.get_benchmark('Mujoco1M')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/ppo/%s/' % timestamp
        result_json = {}
        for task in mujoco1m['tasks']:
            env_id = task['env_id']

            env = gym.make(env_id)
            baseline_env = AutoStopEnv(
                env_name=env_id,
                max_path_length=hyper_parameters['max_path_length'])

            seeds = random.sample(range(100), hyper_parameters['n_trials'])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_id))

            baselines_csvs = []
            garage_tf_csvs = []
            garage_pytorch_csvs = []

            for trial in range(hyper_parameters['n_trials']):
                seed = seeds[trial]

                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                garage_tf_dir = trial_dir + '/garage/tf'
                garage_pytorch_dir = trial_dir + '/garage/pytorch'
                baselines_dir = trial_dir + '/baselines'

                # pylint: disable=not-context-manager
                with tf.Graph().as_default():
                    # Run baselines algorithms
                    baseline_env.reset()
                    baseline_csv = run_baselines(baseline_env, seed,
                                                 baselines_dir)

                    # Run garage algorithms
                    env.reset()
                    garage_tf_csv = run_garage_tf(env, seed, garage_tf_dir)

                env.reset()
                garage_pytorch_csv = run_garage_pytorch(
                    env, seed, garage_pytorch_dir)

                baselines_csvs.append(baseline_csv)
                garage_tf_csvs.append(garage_tf_csv)
                garage_pytorch_csvs.append(garage_pytorch_csv)

            env.close()

            benchmark_helper.plot_average_over_trials(
                [baselines_csvs, garage_tf_csvs, garage_pytorch_csvs],
                ['eprewmean', 'AverageReturn', 'AverageReturn'],
                plt_file=plt_file,
                env_id=env_id,
                x_label='Iteration',
                y_label='AverageReturn',
                names=['baseline', 'garage-TensorFlow', 'garage-PyTorch'],
            )

            result_json[env_id] = benchmark_helper.create_json(
                [baselines_csvs, garage_tf_csvs, garage_pytorch_csvs],
                seeds=seeds,
                trials=hyper_parameters['n_trials'],
                xs=['nupdates', 'Iteration', 'Iteration'],
                ys=['eprewmean', 'AverageReturn', 'AverageReturn'],
                factors=[hyper_parameters['batch_size']] * 3,
                names=['baseline', 'garage-TF', 'garage-PT'])

        Rh.write_file(result_json, 'PPO')


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

    algo = PyTorch_PPO(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       optimizer=torch.optim.Adam,
                       policy_lr=hyper_parameters['learning_rate'],
                       max_path_length=hyper_parameters['max_path_length'],
                       discount=hyper_parameters['discount'],
                       gae_lambda=hyper_parameters['gae_lambda'],
                       center_adv=hyper_parameters['center_adv'],
                       lr_clip_range=hyper_parameters['lr_clip_range'])

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
    """Create garage TensorFlow PPO model and training.

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

        policy = TF_GMP(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['hidden_sizes'],
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TF_PPO(env_spec=env.spec,
                      policy=policy,
                      baseline=baseline,
                      max_path_length=hyper_parameters['max_path_length'],
                      discount=hyper_parameters['discount'],
                      gae_lambda=hyper_parameters['gae_lambda'],
                      center_adv=hyper_parameters['center_adv'],
                      lr_clip_range=hyper_parameters['lr_clip_range'],
                      optimizer_args=dict(
                          batch_size=None,
                          max_epochs=1,
                          tf_optimizer_args=dict(
                              learning_rate=hyper_parameters['learning_rate']),
                          verbose=True))  # yapf: disable

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


def run_baselines(env, seed, log_dir):
    """Create baselines model and training.

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

    env = DummyVecEnv([
        lambda: bench.Monitor(
            env, baselines_logger.get_dir(), allow_early_resets=True)
    ])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy,
               env=env,
               nsteps=hyper_parameters['batch_size'],
               nminibatches=1,
               lam=hyper_parameters['gae_lambda'],
               gamma=hyper_parameters['discount'],
               noptepochs=1,
               log_interval=1,
               ent_coef=0.0,
               vf_coef=0.0,
               max_grad_norm=None,
               lr=hyper_parameters['learning_rate'],
               cliprange=hyper_parameters['lr_clip_range'],
               total_timesteps=hyper_parameters['batch_size'] * hyper_parameters['n_epochs'])  # yapf: disable  # noqa: E501

    return osp.join(log_dir, 'progress.csv')
