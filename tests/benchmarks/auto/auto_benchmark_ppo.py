"""A regression test over PPO Algorithms for automatic benchmarking.

(garage-PyTorch-PPO, garage-TensorFlow-PPO, and baselines-PPO2)
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
from garage.tf.algos import PPO as TF_PPO
from garage.tf.baselines import GaussianMLPBaseline as TF_GMB
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import GaussianMLPPolicy as TF_GMP
from garage.torch.algos import PPO as PyTorch_PPO
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from garage.torch.value_functions import GaussianMLPValueFunction
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
def auto_benchmark_ppo_garage_tf():
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
            ppo_garage_tf.__name__, tasks, seeds):
        ppo_garage_tf(dict(log_dir=log_dir), env_id=env_id, seed=seed)


@pytest.mark.benchmark
def auto_benchmark_ppo_garage_pytorch():
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

        value_function = GaussianMLPValueFunction(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None)

        policy_optimizer = OptimizerWrapper(
            (torch.optim.Adam, dict(lr=2.5e-4)),
            policy,
            max_optimization_epochs=10,
            minibatch_size=64)
        vf_optimizer = OptimizerWrapper((torch.optim.Adam, dict(lr=2.5e-4)),
                                        value_function,
                                        max_optimization_epochs=10,
                                        minibatch_size=64)

        algo = PyTorch_PPO(env_spec=env.spec,
                           policy=policy,
                           value_function=value_function,
                           policy_optimizer=policy_optimizer,
                           vf_optimizer=vf_optimizer,
                           max_path_length=hyper_parameters['max_path_length'],
                           discount=0.99,
                           gae_lambda=0.95,
                           center_adv=True,
                           lr_clip_range=0.2)

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['batch_size'])

    for env_id, seed, log_dir in benchmark_helper.iterate_experiments(
            ppo_garage_pytorch.__name__, tasks, seeds):
        ppo_garage_pytorch(dict(log_dir=log_dir), env_id=env_id, seed=seed)
