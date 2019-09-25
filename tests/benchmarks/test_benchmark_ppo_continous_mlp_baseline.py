'''
This script creates a regression test over garage-DDPG and baselines-DDPG.
It get Mujoco1M benchmarks from baselines benchmark, and test each task in
its trial times on garage model and baselines model. For each task, there will
be `trial` times with different random seeds. For each trial, there will be two
log directories corresponding to baselines and garage. And there will be a plot
plotting the average return curve from baselines and garage.
'''
import datetime
import os
import os.path as osp
import random

from baselines.bench import benchmarks
import dowel
from dowel import logger as dowel_logger
import gym
import pytest
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import deterministic
from garage.tf.algos import PPO
from garage.tf.baselines import (ContinuousMLPBaseline,
                                 ContinuousMLPBaselineWithModel)
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianLSTMPolicy
from tests.fixtures import snapshot_config
# from tests.wrappers import AutoStopEnv

# Hyperparams for baselines and garage

policy_params = {
    'policy_lr': 1e-3,
    'policy_hidden_sizes': 32,
    'hidden_nonlinearity': tf.nn.tanh
}

# baseline_params = {
#     'regressor_args':
#     dict(hidden_sizes=(64, 64),
#          use_trust_region=False,
#          optimizer=FirstOrderOptimizer,
#          optimizer_args=dict(
#              batch_size=32,
#              max_epochs=10,
#              tf_optimizer_args=dict(learning_rate=1e-3),
#          ))  # noqa
# }

baseline_params = {
    'regressor_args': dict(hidden_sizes=(64, 64))
    #  use_trust_region=False,)
}

baseline_with_model_params = {'regressor_args': dict(hidden_sizes=(64, 64))}

algo_params = {
    'n_envs':
    8,
    'n_epochs':
    20,
    'n_rollout_steps':
    2048,
    'discount':
    0.99,
    'max_path_length':
    100,
    'gae_lambda':
    0.95,
    'lr_clip_range':
    0.2,
    'policy_ent_coeff':
    0.02,
    'entropy_method':
    'max',
    'optimizer_args':
    dict(
        batch_size=32,
        max_epochs=10,
        tf_optimizer_args=dict(learning_rate=policy_params['policy_lr']),
    ),
    'center_adv':
    False
}

# number of processing elements to use for tensorflow
num_proc = 4 * 2
# number of trials to run per environment
num_trials = 3


class TestBenchmarkPPOContinuousMLPBaseline:
    '''Compare benchmarks between garage and baselines.'''

    @pytest.mark.huge
    def test_benchmark_ppo_continuous_mlp_baseline(self):
        '''
        Compare benchmarks between CMB with and without Model.
        '''
        mujoco1m = benchmarks.get_benchmark('Mujoco1M')

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = osp.join(os.getcwd(), 'data', 'local', 'benchmarks',
                                 'ppo_cmb', timestamp)
        for task in mujoco1m['tasks']:
            env_id = task['env_id']
            env = gym.make(env_id)

            seeds = random.sample(range(100), num_trials)

            task_dir = osp.join(benchmark_dir, env_id)
            cmb_csvs = []
            cmb_with_model_csvs = []

            for trial in range(num_trials):
                seed = seeds[trial]

                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                cmb_dir = trial_dir + '/continuous_mlp_baseline'
                cmb_with_models_dir = (trial_dir +
                                       '/continuous_mlp_baseline_with_model')

                with tf.Graph().as_default():
                    env.reset()
                    cmb_csv = ppo_cmb(env, seed, cmb_dir)
                with tf.Graph().as_default():
                    env.reset()
                    cmb_with_model_csv = ppo_cmb_w_model(
                        env, seed, cmb_with_models_dir)
                cmb_csvs.append(cmb_csv)
                cmb_with_model_csvs.append(cmb_with_model_csv)

            env.close()


def ppo_cmb(env, seed, log_dir):
    '''
    Create test continuous mlp baseline withOUT model on ppo

    args:
        env - Environment of the task.
        seed - Random seed for the trial.
        log_dir - Log dir path.
    returns:
        tabular_log_file - training results in csv format
    '''
    deterministic.set_seed(seed)
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=num_proc,
                            inter_op_parallelism_threads=num_proc)
    sess = tf.Session(config=config)
    with LocalTFRunner(snapshot_config, sess=sess,
                       max_cpus=num_proc) as runner:
        env = TfEnv(normalize(env))

        policy = GaussianLSTMPolicy(
            env_spec=env.spec,
            hidden_dim=policy_params['policy_hidden_sizes'],
            hidden_nonlinearity=policy_params['hidden_nonlinearity'],
        )

        baseline = ContinuousMLPBaseline(
            env_spec=env.spec,
            regressor_args=baseline_params['regressor_args'],
        )

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   max_path_length=algo_params['max_path_length'],
                   discount=algo_params['discount'],
                   gae_lambda=algo_params['gae_lambda'],
                   lr_clip_range=algo_params['lr_clip_range'],
                   entropy_method=algo_params['entropy_method'],
                   policy_ent_coeff=algo_params['policy_ent_coeff'],
                   optimizer_args=algo_params['optimizer_args'],
                   center_adv=algo_params['center_adv'],
                   stop_entropy_gradient=True)

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo,
                     env,
                     sampler_args=dict(n_envs=algo_params['n_envs']))
        runner.train(n_epochs=algo_params['n_epochs'],
                     batch_size=algo_params['n_rollout_steps'])

        dowel_logger.remove_all()

        return tabular_log_file


def ppo_cmb_w_model(env, seed, log_dir):
    '''
    Create test continuous mlp baseline with model on ppo

    args:
        env - Environment of the task.
        seed - Random seed for the trial.
        log_dir - Log dir path.
    returns:
        tabular_log_file - training results in csv format
    '''
    deterministic.set_seed(seed)
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=num_proc,
                            inter_op_parallelism_threads=num_proc)
    sess = tf.Session(config=config)
    with LocalTFRunner(snapshot_config, sess=sess,
                       max_cpus=num_proc) as runner:
        env = TfEnv(normalize(env))

        policy = GaussianLSTMPolicy(
            env_spec=env.spec,
            hidden_dim=policy_params['policy_hidden_sizes'],
            hidden_nonlinearity=policy_params['hidden_nonlinearity'],
        )

        baseline = ContinuousMLPBaselineWithModel(
            env_spec=env.spec,
            regressor_args=baseline_with_model_params['regressor_args'],
        )

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   max_path_length=algo_params['max_path_length'],
                   discount=algo_params['discount'],
                   gae_lambda=algo_params['gae_lambda'],
                   lr_clip_range=algo_params['lr_clip_range'],
                   entropy_method=algo_params['entropy_method'],
                   policy_ent_coeff=algo_params['policy_ent_coeff'],
                   optimizer_args=algo_params['optimizer_args'],
                   center_adv=algo_params['center_adv'],
                   stop_entropy_gradient=True)

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo,
                     env,
                     sampler_args=dict(n_envs=algo_params['n_envs']))
        runner.train(n_epochs=algo_params['n_epochs'],
                     batch_size=algo_params['n_rollout_steps'])

        dowel_logger.remove_all()

        return tabular_log_file
