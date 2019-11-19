"""Benchmark categorical gru policy."""
import datetime
import os.path as osp
import random

import dowel
from dowel import logger as dowel_logger
import gym
import pytest
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import deterministic
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalGRUPolicy
from tests.fixtures import snapshot_config
import tests.helpers as Rh


@pytest.mark.huge
def test_benchmark_categorical_gru_policy():
    """Benchmark categorical gru policy."""
    categorical_tasks = [
        'LunarLander-v2',
        'Assault-ramDeterministic-v4',
        'Breakout-ramDeterministic-v4',
        'ChopperCommand-ramDeterministic-v4',
        'Tutankham-ramDeterministic-v4',
        'CartPole-v1',
    ]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    benchmark_dir = './data/local/benchmarks/ppo_categ_gru/%s/' % timestamp
    result_json = {}
    for task in categorical_tasks:
        env_id = task
        env = gym.make(env_id)
        # baseline_env = AutoStopEnv(env_name=env_id, max_path_length=100)

        seeds = random.sample(range(100), 3)

        task_dir = osp.join(benchmark_dir, env_id)
        plt_file = osp.join(benchmark_dir, '{}_benchmark.png'.format(env_id))
        baselines_csvs = []
        garage_csvs = []

        for trial in range(3):
            seed = seeds[trial]

            trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
            garage_dir = trial_dir + '/garage'

            with tf.Graph().as_default():
                # Run baselines algorithms
                # baseline_env.reset()
                # baselines_csv = run_baselines(baseline_env, seed,
                #                               baselines_dir)

                # Run garage algorithms
                env.reset()
                garage_csv = run_garage(env, seed, garage_dir)

            garage_csvs.append(garage_csv)

        env.close()

        Rh.plot(b_csvs=baselines_csvs,
                g_csvs=garage_csvs,
                g_x='Iteration',
                g_y='AverageReturn',
                g_z='garage',
                b_x='Iteration',
                b_y='AverageReturn',
                b_z='baselines',
                trials=3,
                seeds=seeds,
                plt_file=plt_file,
                env_id=env_id,
                x_label='Iteration',
                y_label='AverageReturn')

        result_json[env_id] = Rh.create_json(b_csvs=baselines_csvs,
                                             g_csvs=garage_csvs,
                                             seeds=seeds,
                                             trails=3,
                                             g_x='Iteration',
                                             g_y='AverageReturn',
                                             b_x='Iteration',
                                             b_y='AverageReturn',
                                             factor_g=2048,
                                             factor_b=2048)

    Rh.write_file(result_json, 'PPO')


def run_garage(env, seed, log_dir):
    """Create garage model and training.

    Replace the ppo with the algorithm you want to run.

    Args:
        env (gym.Env): Environment of the task.
        seed (int): Random seed for the trial.
        log_dir (str): Log dir path.
    Returns:
        str: Path to output csv file
    """
    deterministic.set_seed(seed)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                      intra_op_parallelism_threads=12,
                                      inter_op_parallelism_threads=12)
    sess = tf.compat.v1.Session(config=config)
    with LocalTFRunner(snapshot_config, sess=sess, max_cpus=12) as runner:
        env = TfEnv(normalize(env))

        policy = CategoricalGRUPolicy(
            env_spec=env.spec,
            hidden_dim=32,
            hidden_nonlinearity=tf.nn.tanh,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            policy_ent_coeff=0.0,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                tf_optimizer_args=dict(learning_rate=1e-3),
            ),
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo, env, sampler_args=dict(n_envs=12))
        runner.train(n_epochs=488, batch_size=2048)
        dowel_logger.remove_all()

        return tabular_log_file
