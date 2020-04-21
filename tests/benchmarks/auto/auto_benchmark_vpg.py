"""A regression test over VPG algorithms.
(garage-PyTorch-VPG, garage-TensorFlow-VPG)
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
from garage.tf.algos import VPG as TF_VPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy as TF_GMP
from garage.torch.algos import VPG as PyTorch_VPG
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from tests import benchmark_helper

hyper_parameters = {
    'hidden_sizes': [64, 64],
    'center_adv': True,
    'learning_rate': 1e-2,
    'discount': 0.99,
    'n_epochs': 250,
    'max_path_length': 100,
    'batch_size': 2048,
    'n_trials': 10
}

seeds = random.sample(range(100), hyper_parameters['n_trials'])
tasks = benchmarks.get_benchmark('Mujoco1M')['tasks']


@pytest.mark.benchmark
def auto_benchmark_vpg_garage_tf():
    """Create garage TensorFlow VPG model and training.

    Training over different environments and seeds.

    """

    @wrap_experiment
    def vpg_garage_tf(ctxt, env_id, seed):
        """Create garage TensorFlow VPG model and training.

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
                hidden_sizes=hyper_parameters['hidden_sizes'],
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=None,
            )

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TF_VPG(env_spec=env.spec,
                          policy=policy,
                          baseline=baseline,
                          max_path_length=hyper_parameters['max_path_length'],
                          discount=hyper_parameters['discount'],
                          center_adv=hyper_parameters['center_adv'],
                          optimizer_args=dict(tf_optimizer_args=dict(
                              learning_rate=hyper_parameters['learning_rate']),
                                              verbose=True))

            runner.setup(algo, env)
            runner.train(n_epochs=hyper_parameters['n_epochs'],
                         batch_size=hyper_parameters['batch_size'])

    for env_id, seed, log_dir in benchmark_helper.iterate_experiments(
            vpg_garage_tf, tasks, seeds):
        vpg_garage_tf(dict(log_dir=log_dir), env_id=env_id, seed=seed)


@pytest.mark.benchmark
def auto_benchmark_vpg_garage_pytorch():
    """Create garage PyTorch VPG model and training.

    Training over different environments and seeds.

    """

    @wrap_experiment
    def vpg_garage_pytorch(ctxt, env_id, seed):
        """Create garage PyTorch VPG model and training.

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

        algo = PyTorch_VPG(env_spec=env.spec,
                           policy=policy,
                           optimizer=torch.optim.Adam,
                           policy_lr=hyper_parameters['learning_rate'],
                           value_function=value_function,
                           max_path_length=hyper_parameters['max_path_length'],
                           discount=hyper_parameters['discount'],
                           center_adv=hyper_parameters['center_adv'])

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['batch_size'])

    for env_id, seed, log_dir in benchmark_helper.iterate_experiments(
            vpg_garage_pytorch, tasks, seeds):
        vpg_garage_pytorch(dict(log_dir=log_dir), env_id=env_id, seed=seed)
