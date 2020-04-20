"""A regression test over TRPO Algorithms for automatic benchmarking.
(garage-PyTorch-TRPO, garage-TensorFlow-TRPO, and baselines-TRPO)

Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length. We also need to change the
garage.tf.samplers.BatchSampler to smooth the reward curve.
"""
import random

from baselines.bench import benchmarks
import gym
import pytest
import tensorflow as tf
import torch

from garage import wrap_experiment
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

seeds = random.sample(range(100), hyper_parameters['n_trials'])
tasks = benchmarks.get_benchmark('Mujoco1M')['tasks']


@pytest.mark.benchmark
def auto_benchmark_trpo_garage_tf():
    """Create garage TensorFlow TRPO model and training.

    Training over different environments and seeds.

    """

    @wrap_experiment
    def trpo_garage_tf(ctxt, env_id, seed):
        """Create garage Tensorflow TROI model and training.

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

            runner.setup(algo, env)
            runner.train(n_epochs=hyper_parameters['n_epochs'],
                         batch_size=hyper_parameters['batch_size'])

    for env_id, seed, log_dir in benchmark_helper.iterate_experiments(
            trpo_garage_tf.__name__, tasks, seeds):
        trpo_garage_tf(dict(log_dir=log_dir), env_id=env_id, seed=seed)


@pytest.mark.benchmark
def auto_benchmark_trpo_garage_pytorch():
    """Create garage PyTorch TRPO model and training.

    Training over different environments and seeds.

    """

    @wrap_experiment
    def trpo_garage_pytorch(ctxt, env_id, seed):
        """Create garage PyTorch TRPO model and training.

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
                             hidden_sizes=hyper_parameters['hidden_sizes'],
                             hidden_nonlinearity=torch.tanh,
                             output_nonlinearity=None)

        value_function = LinearFeatureBaseline(env_spec=env.spec)

        algo = PyTorch_TRPO(
            env_spec=env.spec,
            policy=policy,
            value_function=value_function,
            max_kl_step=hyper_parameters['max_kl'],
            max_path_length=hyper_parameters['max_path_length'],
            discount=hyper_parameters['discount'],
            gae_lambda=hyper_parameters['gae_lambda'])

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['batch_size'])

    for env_id, seed, log_dir in benchmark_helper.iterate_experiments(
            trpo_garage_pytorch.__name__, tasks, seeds):
        trpo_garage_pytorch(dict(log_dir=log_dir), env_id=env_id, seed=seed)
