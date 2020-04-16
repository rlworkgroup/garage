"""A regression test over PPO Algorithms for automatic benchmarking.
(garage-PyTorch-PPO, garage-TensorFlow-PPO, and baselines-PPO2)
"""
import multiprocessing
import random

from baselines import bench
from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.logger import configure
from baselines.ppo2 import ppo2
import gym
import pytest
import tensorflow as tf
import torch

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic, LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO as TF_PPO
from garage.tf.baselines import GaussianMLPBaseline as TF_GMB
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import GaussianMLPPolicy as TF_GMP
from garage.torch.algos import PPO as PyTorch_PPO
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from tests import benchmark_helper

hyper_parameters = {
    'n_epochs': 800,
    'max_path_length': 128,
    'batch_size': 1024,
    'n_trials': 4,
}

seeds = random.sample(range(100), hyper_parameters['n_trials'])
tasks = benchmarks.get_benchmark('Mujoco1M')['tasks']


@pytest.mark.benchmark
def test_auto_benchmark_ppo_garage_tf():
    """Create garage TensorFlow PPO model and training.

    Training over different environments and seeds.

    """

    @wrap_experiment
    def ppo_garage_tf(ctxt, env_id, seed):
        """Create garage TensorFlow PPO model and training.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by LocalRunner to create the
                snapshotter.
            env_id (str): Environment id of the task.
            seed (int): Random positive integer for the trial.

        """
        deterministic.set_seed(seed)

        with LocalTFRunner(ctxt) as runner:
            env = TfEnv(normalize(gym.make(env_id)))

            policy = TF_GMP(
                env_spec=env.spec,
                hidden_sizes=(32, 32),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=None,
            )

            baseline = TF_GMB(
                env_spec=env.spec,
                regressor_args=dict(
                    hidden_sizes=(32, 32),
                    use_trust_region=False,
                    optimizer=FirstOrderOptimizer,
                    optimizer_args=dict(
                        batch_size=32,
                        max_epochs=10,
                        tf_optimizer_args=dict(learning_rate=3e-4),
                    ),
                ),
            )

            algo = TF_PPO(env_spec=env.spec,
                          policy=policy,
                          baseline=baseline,
                          max_path_length=hyper_parameters['max_path_length'],
                          discount=0.99,
                          gae_lambda=0.95,
                          center_adv=True,
                          lr_clip_range=0.2,
                          optimizer_args=dict(
                              batch_size=32,
                              max_epochs=10,
                              tf_optimizer_args=dict(learning_rate=3e-4),
                              verbose=True))

            runner.setup(algo, env)
            runner.train(n_epochs=hyper_parameters['n_epochs'],
                         batch_size=hyper_parameters['batch_size'])

    for env_id, seed, log_dir in benchmark_helper.iterate_experiments(
            ppo_garage_tf,
            tasks,
            seeds,
            use_tf=True,
            xcolumn='TotalEnvSteps',
            xlabel='Total Environment Steps',
            ycolumn='Evaluation/AverageReturn',
            ylabel='Average Return'):
        ppo_garage_tf(dict(log_dir=log_dir), env_id=env_id, seed=seed)


@pytest.mark.benchmark
def test_auto_benchmark_ppo_baselines():
    """Create baselines model and training.

    Training over different environments and seeds.

    """

    def ppo_baselines(log_dir, env_id, seed):
        """Create baselines model and training.

        Args:
            log_dir (str): Experiment log directory.
            env_id (str): Environment id of the task.
            seed (int): Random positive integer for the trial.

        """
        # Set up TF Session
        ncpu = max(multiprocessing.cpu_count() // 2, 1)
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=ncpu,
                                inter_op_parallelism_threads=ncpu)
        tf.compat.v1.Session(config=config).__enter__()

        # Set up baselines logger
        configure(dir=log_dir,
                  format_strs=['stdout', 'log', 'csv', 'tensorboard'])
        baselines_logger.info('rank {}: seed={}, logdir={}'.format(
            0, seed, baselines_logger.get_dir()))

        set_global_seeds(seed)

        env = DummyVecEnv([
            lambda: bench.Monitor(gym.make(env_id),
                                  baselines_logger.get_dir(),
                                  allow_early_resets=True)
        ])

        ppo2.learn(network='mlp',
                   env=env,
                   nsteps=hyper_parameters['batch_size'],
                   nminibatches=32,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=10,
                   log_interval=1,
                   ent_coef=0.0,
                   max_grad_norm=None,
                   lr=3e-4,
                   cliprange=0.2,
                   total_timesteps=(hyper_parameters['batch_size'] *
                                    hyper_parameters['n_epochs']))

    for env_id, seed, log_dir in benchmark_helper.iterate_experiments(
            ppo_baselines,
            tasks,
            seeds,
            use_tf=True,
            xcolumn='misc/total_timesteps',
            xlabel='Total Environment Steps',
            ycolumn='eprewmean',
            ylabel='Average Return'):
        ppo_baselines(log_dir=log_dir, env_id=env_id, seed=seed)


@pytest.mark.benchmark
def test_auto_benchmark_ppo_garage_pytorch():
    """Create garage PyTorch PPO model and training.

    Training over different environments and seeds.

    """

    @wrap_experiment
    def ppo_garage_pytorch(ctxt, env_id, seed):
        """Create garage PyTorch PPO model and training.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by LocalRunner to create the
                snapshotter.
            env_id (str): Environment id of the task.
            seed (int): Random positive integer for the trial.

        """
        deterministic.set_seed(seed)

        runner = LocalRunner(ctxt)

        env = TfEnv(normalize(gym.make(env_id)))

        policy = PyTorch_GMP(env.spec,
                             hidden_sizes=(32, 32),
                             hidden_nonlinearity=torch.tanh,
                             output_nonlinearity=None)

        value_functions = LinearFeatureBaseline(env_spec=env.spec)

        algo = PyTorch_PPO(env_spec=env.spec,
                           policy=policy,
                           value_function=value_functions,
                           optimizer=torch.optim.Adam,
                           policy_lr=3e-4,
                           max_path_length=hyper_parameters['max_path_length'],
                           discount=0.99,
                           gae_lambda=0.95,
                           center_adv=True,
                           lr_clip_range=0.2,
                           minibatch_size=128,
                           max_optimization_epochs=10)

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['batch_size'])

    for env_id, seed, log_dir in benchmark_helper.iterate_experiments(
            ppo_garage_pytorch,
            tasks,
            seeds,
            use_tf=False,
            xcolumn='TotalEnvSteps',
            xlabel='Total Environment Steps',
            ycolumn='Evaluation/AverageReturn',
            ylabel='Average Return'):
        ppo_garage_pytorch(dict(log_dir=log_dir), env_id=env_id, seed=seed)
