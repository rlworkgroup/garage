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

from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common.misc_util import set_global_seeds
from baselines.ddpg.memory import Memory
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
import baselines.ddpg.training as training
from baselines.logger import configure
import dowel
from dowel import logger as dowel_logger
import gym
from mpi4py import MPI
import numpy as np
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
    'n_epoch_cycles': 20,
    'n_rollout_steps': 100,
    'n_train_steps': 50,
    'discount': 0.9,
    'tau': 1e-2,
    'replay_buffer_size': int(1e6),
    'sigma': 0.2,
}


class TestBenchmarkDDPG:
    '''Compare benchmarks between garage and baselines.'''

    @pytest.mark.huge
    def test_benchmark_ddpg(self):
        '''
        Compare benchmarks between garage and baselines.
        :return:
        '''
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
            baseline_env = AutoStopEnv(
                env_name=env_id, max_path_length=params['n_rollout_steps'])
            seeds = random.sample(range(100), task['trials'])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_id))
            relplt_file = osp.join(benchmark_dir,
                                   '{}_benchmark_mean.png'.format(env_id))
            baselines_csvs = []
            garage_csvs = []

            for trial in range(task['trials']):
                env.reset()
                baseline_env.reset()
                seed = seeds[trial]

                trial_dir = osp.join(
                    task_dir, 'trial_{}_seed_{}'.format(trial + 1, seed))
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
                    g_y='AverageReturn',
                    g_z='Garage',
                    b_x='total/epochs',
                    b_y='rollout/return',
                    b_z='Baseline',
                    trials=task['trials'],
                    seeds=seeds,
                    plt_file=plt_file,
                    env_id=env_id,
                    x_label='Epoch',
                    y_label='AverageReturn')

            Rh.relplot(g_csvs=garage_csvs,
                       b_csvs=baselines_csvs,
                       g_x='Epoch',
                       g_y='AverageReturn',
                       g_z='Garage',
                       b_x='total/epochs',
                       b_y='rollout/return',
                       b_z='Baseline',
                       trials=task['trials'],
                       seeds=seeds,
                       plt_file=relplt_file,
                       env_id=env_id,
                       x_label='Epoch',
                       y_label='AverageReturn')

            result_json[env_id] = Rh.create_json(
                b_csvs=baselines_csvs,
                g_csvs=garage_csvs,
                seeds=seeds,
                trails=task['trials'],
                g_x='Epoch',
                g_y='AverageReturn',
                b_x='total/epochs',
                b_y='rollout/return',
                factor_g=params['n_epoch_cycles'] * params['n_rollout_steps'],
                factor_b=1)

        Rh.write_file(result_json, 'DDPG')


def run_garage(env, seed, log_dir):
    '''
    Create garage model and training.
    Replace the ddpg with the algorithm you want to run.
    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return:
    '''
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

        ddpg = DDPG(env_spec=env.spec,
                    policy=policy,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    policy_lr=params['policy_lr'],
                    qf_lr=params['qf_lr'],
                    target_update_tau=params['tau'],
                    n_train_steps=params['n_train_steps'],
                    discount=params['discount'],
                    min_buffer_size=int(1e4),
                    exploration_strategy=action_noise,
                    policy_optimizer=tf.train.AdamOptimizer,
                    qf_optimizer=tf.train.AdamOptimizer)

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        tensorboard_log_dir = osp.join(log_dir)
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(tensorboard_log_dir))

        runner.setup(ddpg, env)
        runner.train(n_epochs=params['n_epochs'],
                     n_epoch_cycles=params['n_epoch_cycles'],
                     batch_size=params['n_rollout_steps'])

        dowel_logger.remove_all()

        return tabular_log_file


def run_baselines(env, seed, log_dir):
    '''
    Create baselines model and training.
    Replace the ddpg and its training with the algorithm you want to run.
    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return
    '''
    rank = MPI.COMM_WORLD.Get_rank()
    seed = seed + 1000000 * rank
    set_global_seeds(seed)
    env.seed(seed)

    # Set up logger for baselines
    configure(dir=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    baselines_logger.info('rank {}: seed={}, logdir={}'.format(
        rank, seed, baselines_logger.get_dir()))

    # Set up params for baselines ddpg
    nb_actions = env.action_space.shape[-1]
    layer_norm = False

    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                sigma=float(params['sigma']) *
                                                np.ones(nb_actions))
    memory = Memory(limit=params['replay_buffer_size'],
                    action_shape=env.action_space.shape,
                    observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    training.train(env=env,
                   eval_env=None,
                   param_noise=None,
                   action_noise=action_noise,
                   actor=actor,
                   critic=critic,
                   memory=memory,
                   nb_epochs=params['n_epochs'],
                   nb_epoch_cycles=params['n_epoch_cycles'],
                   render_eval=False,
                   reward_scale=1.,
                   render=False,
                   normalize_returns=False,
                   normalize_observations=False,
                   critic_l2_reg=0,
                   actor_lr=params['policy_lr'],
                   critic_lr=params['qf_lr'],
                   popart=False,
                   gamma=params['discount'],
                   clip_norm=None,
                   nb_train_steps=params['n_train_steps'],
                   nb_rollout_steps=params['n_rollout_steps'],
                   nb_eval_steps=100,
                   batch_size=64)

    return osp.join(log_dir, 'progress.csv')
