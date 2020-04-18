"""This script creates a regression test over garage-DDPG and baselines-DDPG.
It get Mujoco1M benchmarks from baselines benchmark, and test each task in
its trial times on garage model and baselines model. For each task, there will
be `trial` times with different random seeds. For each trial, there will be two
log directories corresponding to baselines and garage. And there will be a plot
plotting the average return curve from baselines and garage.
"""
import datetime
import os
import os.path as osp
import random

from baselines import bench
from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common.misc_util import set_global_seeds
from baselines.common.vec_env import DummyVecEnv
from baselines.ddpg import ddpg
from baselines.logger import configure
import dowel
from dowel import logger as dowel_logger
import gym
import pytest
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import deterministic
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from tests.fixtures import snapshot_config
import tests.helpers as Rh
from tests.wrappers import AutoStopEnv

# Hyperparams for baselines and garage
params = {
    'policy_lr': 1e-4,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [64, 64],
    'qf_hidden_sizes': [64, 64],
    'n_epochs': 500,
    'steps_per_epoch': 20,
    'n_rollout_steps': 100,
    'n_train_steps': 50,
    'discount': 0.9,
    'tau': 1e-2,
    'replay_buffer_size': int(1e6),
    'sigma': 0.2,
}


def benchmark_ddpg():
    """Compare benchmarks between garage and baselines."""
    # Load Mujoco1M tasks, you can check other benchmarks here
    # https://github.com/openai/baselines/blob/master/baselines/bench/benchmarks.py
    mujoco1m = benchmarks.get_benchmark('Mujoco1M')

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    benchmark_dir = osp.join(os.getcwd(), 'data', 'local', 'benchmarks',
                             'ddpg', timestamp)
    result_json = {}
    for task in mujoco1m['tasks']:
        env_id = task['env_id']
        env = gym.make(env_id)
        baseline_env = AutoStopEnv(env_name=env_id,
                                   max_path_length=params['n_rollout_steps'])
        seeds = random.sample(range(100), task['trials'])

        task_dir = osp.join(benchmark_dir, env_id)
        plt_file = osp.join(benchmark_dir, '{}_benchmark.png'.format(env_id))
        relplt_file = osp.join(benchmark_dir,
                               '{}_benchmark_mean.png'.format(env_id))
        baselines_csvs = []
        garage_csvs = []

        for trial in range(task['trials']):
            env.reset()
            baseline_env.reset()
            seed = seeds[trial]

            trial_dir = osp.join(task_dir,
                                 'trial_{}_seed_{}'.format(trial + 1, seed))
            garage_dir = osp.join(trial_dir, 'garage')
            baselines_dir = osp.join(trial_dir, 'baselines')

            with tf.Graph().as_default():
                # Run garage algorithms
                garage_csv = run_garage(env, seed, garage_dir)

                # Run baselines algorithms
                baselines_csv = run_baselines(baseline_env, seed,
                                              baselines_dir)

            garage_csvs.append(garage_csv)
            baselines_csvs.append(baselines_csv)

        env.close()

        Rh.plot(b_csvs=baselines_csvs,
                g_csvs=garage_csvs,
                g_x='Epoch',
                g_y='Evaluation/AverageReturn',
                g_z='Garage',
                b_x='total/epochs',
                b_y='rollout/return',
                b_z='Baseline',
                trials=task['trials'],
                seeds=seeds,
                plt_file=plt_file,
                env_id=env_id,
                x_label='Epoch',
                y_label='Evaluation/AverageReturn')

        Rh.relplot(g_csvs=garage_csvs,
                   b_csvs=baselines_csvs,
                   g_x='Epoch',
                   g_y='Evaluation/AverageReturn',
                   g_z='Garage',
                   b_x='total/epochs',
                   b_y='rollout/return',
                   b_z='Baseline',
                   trials=task['trials'],
                   seeds=seeds,
                   plt_file=relplt_file,
                   env_id=env_id,
                   x_label='Epoch',
                   y_label='Evaluation/AverageReturn')

        result_json[env_id] = Rh.create_json(
            b_csvs=baselines_csvs,
            g_csvs=garage_csvs,
            seeds=seeds,
            trails=task['trials'],
            g_x='Epoch',
            g_y='Evaluation/AverageReturn',
            b_x='total/epochs',
            b_y='rollout/return',
            factor_g=params['steps_per_epoch'] * params['n_rollout_steps'],
            factor_b=1)

    Rh.write_file(result_json, 'DDPG')


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

    with LocalTFRunner(snapshot_config) as runner:
        env = TfEnv(normalize(env))
        # Set up params for ddpg
        action_noise = OUStrategy(env.spec, sigma=params['sigma'])

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=params['policy_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=params['qf_hidden_sizes'],
                                    hidden_nonlinearity=tf.nn.relu)

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=params['replay_buffer_size'],
            time_horizon=params['n_rollout_steps'])

        algo = DDPG(env_spec=env.spec,
                    policy=policy,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    steps_per_epoch=params['steps_per_epoch'],
                    policy_lr=params['policy_lr'],
                    qf_lr=params['qf_lr'],
                    target_update_tau=params['tau'],
                    n_train_steps=params['n_train_steps'],
                    discount=params['discount'],
                    min_buffer_size=int(1e4),
                    exploration_strategy=action_noise,
                    policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                    qf_optimizer=tf.compat.v1.train.AdamOptimizer)

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        tensorboard_log_dir = osp.join(log_dir)
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(tensorboard_log_dir))

        runner.setup(algo, env)
        runner.train(n_epochs=params['n_epochs'],
                     batch_size=params['n_rollout_steps'])

        dowel_logger.remove_all()

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
    seed = seed + 1000000
    set_global_seeds(seed)
    env.seed(seed)

    # Set up logger for baselines
    configure(dir=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    baselines_logger.info('seed={}, logdir={}'.format(
        seed, baselines_logger.get_dir()))

    env = DummyVecEnv([
        lambda: bench.Monitor(
            env, baselines_logger.get_dir(), allow_early_resets=True)
    ])

    ddpg.learn(network='mlp',
               env=env,
               nb_epochs=params['n_epochs'],
               nb_epoch_cycles=params['steps_per_epoch'],
               normalize_observations=False,
               critic_l2_reg=0,
               actor_lr=params['policy_lr'],
               critic_lr=params['qf_lr'],
               gamma=params['discount'],
               nb_train_steps=params['n_train_steps'],
               nb_rollout_steps=params['n_rollout_steps'],
               nb_eval_steps=100)

    return osp.join(log_dir, 'progress.csv')
