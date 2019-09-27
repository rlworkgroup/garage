"""
This script creates a regression test over garage-TD3.

It get Mujoco1M benchmarks from baselines benchmark, and test each task in
its trail times on garage model and rlkit model. For each task, there will
be `trail` times with different random seeds. For each trail, there will be two
log directories corresponding to rlkit and garage. And there will be a plot
plotting the average return curve from rlkit and garage.
"""
import datetime
import os
import os.path as osp
import random

from baselines.bench import benchmarks
import dowel
from dowel import logger as dowel_logger
import gtimer as gt
import gym
import pytest
try:
    from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
    from rlkit.envs.wrappers import NormalizedBoxEnv
    from rlkit.exploration_strategies.base import \
        PolicyWrappedWithExplorationStrategy
    from rlkit.exploration_strategies.gaussian_strategy import \
        GaussianStrategy as RLkitGaussianStrategy
    from rlkit.launchers.launcher_util import reset_execution_environment
    from rlkit.launchers.launcher_util import setup_logger
    from rlkit.samplers.data_collector import MdpPathCollector
    from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.td3.td3 import TD3Trainer
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
except ImportError:
    print('Error: rlkit not found. You can install it by running '
          '`git clone https://github.com/vitchyr/rlkit`'
          '`cd rlkit && pip install -e .`')
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import deterministic
from garage.np.exploration_strategies.gaussian_strategy import GaussianStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import TD3
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from tests.fixtures import snapshot_config
import tests.helpers as Rh
from tests.wrappers import AutoStopEnv

# Hyperparams for rlkit and garage
params = {
    'policy_lr': 1e-3,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [400, 300],
    'qf_hidden_sizes': [400, 300],
    'n_epochs': 1000,
    'n_epoch_cycles': 20,
    'n_rollout_steps': 250,
    'n_train_steps': 1,
    'discount': 0.99,
    'tau': 0.005,
    'replay_buffer_size': int(1e6),
    'sigma': 0.1,
    'smooth_return': False,
    'buffer_batch_size': 100,
    'min_buffer_size': int(1e4)
}


class TestBenchmarkTD3:
    """Benchmark Garage TD3 implementation with rlkit's implementation."""

    @pytest.mark.huge
    def test_benchmark_td3(self):
        """
        Test garage TD3 benchmarks.

        :return:
        """
        # Load Mujoco1M tasks, you can check other benchmarks here
        # https://github.com/openai/baselines/blob/master/baselines/bench/benchmarks.py # noqa: E501
        mujoco1m = benchmarks.get_benchmark('Mujoco1M')

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = osp.join(os.getcwd(), 'data', 'local', 'benchmarks',
                                 'td3', timestamp)
        result_json = {}

        # rlkit throws error for'Reacher-V2' due to gym version mismatch
        mujoco1m['tasks'] = [
            task for task in mujoco1m['tasks']
            if task['env_id'] != 'Reacher-v2'
        ]

        for task in mujoco1m['tasks']:
            env_id = task['env_id']
            env = gym.make(env_id)
            rlkit_env = AutoStopEnv(env_name=env_id,
                                    max_path_length=params['n_rollout_steps'])
            seeds = random.sample(range(100), task['trials'])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_id))
            garage_csvs = []
            rlkit_csvs = []

            for trial in range(task['trials']):
                env.reset()
                rlkit_env.reset()
                seed = seeds[trial]

                trial_dir = osp.join(
                    task_dir, 'trial_{}_seed_{}'.format(trial + 1, seed))
                garage_dir = trial_dir + '/garage'
                rlkit_dir = osp.join(trial_dir, 'rlkit')

                with tf.Graph().as_default():
                    # Run rlkit algorithms
                    rlkit_csv = run_rlkit(rlkit_env, seed, rlkit_dir)

                    # Run garage algorithms
                    garage_csv = run_garage(env, seed, garage_dir)

                garage_csvs.append(garage_csv)
                rlkit_csvs.append(rlkit_csv)

            Rh.plot(b_csvs=rlkit_csvs,
                    g_csvs=garage_csvs,
                    g_x='Epoch',
                    g_y='AverageReturn',
                    g_z='garage',
                    b_x='Epoch',
                    b_y='evaluation/Average Returns',
                    b_z='rlkit',
                    trials=task['trials'],
                    seeds=seeds,
                    plt_file=plt_file,
                    env_id=env_id,
                    x_label='Iteration',
                    y_label='AverageReturn')

            result_json[env_id] = Rh.create_json(
                b_csvs=rlkit_csvs,
                g_csvs=garage_csvs,
                seeds=seeds,
                trails=task['trials'],
                g_x='Epoch',
                g_y='AverageReturn',
                b_x='Epoch',
                b_y='evaluation/Average Returns',
                factor_g=1,
                factor_b=1)

        Rh.write_file(result_json, 'TD3')

    test_benchmark_td3.huge = True


def run_garage(env, seed, log_dir):
    """
    Create garage model and training.

    Replace the td3 with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return:
    """
    deterministic.set_seed(seed)

    with LocalTFRunner(snapshot_config) as runner:
        env = TfEnv(normalize(env))
        # Set up params for TD3
        exploration_noise = GaussianStrategy(env.spec,
                                             max_sigma=params['sigma'],
                                             min_sigma=params['sigma'])

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=params['policy_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        qf = ContinuousMLPQFunction(name='ContinuousMLPQFunction',
                                    env_spec=env.spec,
                                    hidden_sizes=params['qf_hidden_sizes'],
                                    action_merge_layer=0,
                                    hidden_nonlinearity=tf.nn.relu)

        qf2 = ContinuousMLPQFunction(name='ContinuousMLPQFunction2',
                                     env_spec=env.spec,
                                     hidden_sizes=params['qf_hidden_sizes'],
                                     action_merge_layer=0,
                                     hidden_nonlinearity=tf.nn.relu)

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=params['replay_buffer_size'],
            time_horizon=params['n_rollout_steps'])

        td3 = TD3(env.spec,
                  policy=policy,
                  qf=qf,
                  qf2=qf2,
                  replay_buffer=replay_buffer,
                  policy_lr=params['policy_lr'],
                  qf_lr=params['qf_lr'],
                  target_update_tau=params['tau'],
                  n_epoch_cycles=params['n_epoch_cycles'],
                  n_train_steps=params['n_train_steps'],
                  discount=params['discount'],
                  smooth_return=params['smooth_return'],
                  min_buffer_size=params['min_buffer_size'],
                  buffer_batch_size=params['buffer_batch_size'],
                  exploration_strategy=exploration_noise,
                  policy_optimizer=tf.train.AdamOptimizer,
                  qf_optimizer=tf.train.AdamOptimizer)

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(td3, env)
        runner.train(n_epochs=params['n_epochs'],
                     batch_size=params['n_rollout_steps'],
                     n_epoch_cycles=params['n_epoch_cycles'])

        dowel_logger.remove_all()

        return tabular_log_file


def run_rlkit(env, seed, log_dir):
    """
    Create rlkit model and training.

    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return result csv file
    """
    reset_execution_environment()
    gt.reset()
    setup_logger(log_dir=log_dir)

    expl_env = NormalizedBoxEnv(env)
    eval_env = NormalizedBoxEnv(env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    qf1 = FlattenMlp(input_size=obs_dim + action_dim,
                     output_size=1,
                     hidden_sizes=params['qf_hidden_sizes'])
    qf2 = FlattenMlp(input_size=obs_dim + action_dim,
                     output_size=1,
                     hidden_sizes=params['qf_hidden_sizes'])
    target_qf1 = FlattenMlp(input_size=obs_dim + action_dim,
                            output_size=1,
                            hidden_sizes=params['qf_hidden_sizes'])
    target_qf2 = FlattenMlp(input_size=obs_dim + action_dim,
                            output_size=1,
                            hidden_sizes=params['qf_hidden_sizes'])
    policy = TanhMlpPolicy(input_size=obs_dim,
                           output_size=action_dim,
                           hidden_sizes=params['policy_hidden_sizes'])
    target_policy = TanhMlpPolicy(input_size=obs_dim,
                                  output_size=action_dim,
                                  hidden_sizes=params['policy_hidden_sizes'])
    es = RLkitGaussianStrategy(
        action_space=expl_env.action_space,
        max_sigma=params['sigma'],
        min_sigma=params['sigma'],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        exploration_policy,
    )
    replay_buffer = EnvReplayBuffer(
        params['replay_buffer_size'],
        expl_env,
    )
    trainer = TD3Trainer(policy=policy,
                         qf1=qf1,
                         qf2=qf2,
                         target_qf1=target_qf1,
                         target_qf2=target_qf2,
                         target_policy=target_policy,
                         discount=params['discount'])
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        num_epochs=params['n_epochs'],
        num_train_loops_per_epoch=params['n_epoch_cycles'],
        num_trains_per_train_loop=params['n_train_steps'],
        num_expl_steps_per_train_loop=params['n_rollout_steps'],
        num_eval_steps_per_epoch=params['n_rollout_steps'],
        min_num_steps_before_training=params['min_buffer_size'],
        max_path_length=params['n_rollout_steps'],
        batch_size=params['buffer_batch_size'],
    )
    algorithm.to(ptu.device)
    algorithm.train()
    return osp.join(log_dir, 'progress.csv')
