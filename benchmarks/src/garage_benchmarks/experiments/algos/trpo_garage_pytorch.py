"""A regression test for automatic benchmarking garage-PyTorch-TRPO."""
import gym
import torch

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.torch.algos import TRPO as PyTorch_TRPO
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from garage.torch.value_functions import GaussianMLPValueFunction

hyper_parameters = {
    'hidden_sizes': [32, 32],
    'max_kl': 0.01,
    'gae_lambda': 0.97,
    'discount': 0.99,
    'max_path_length': 100,
    'n_epochs': 999,
    'batch_size': 1024,
}


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

    env = GarageEnv(normalize(gym.make(env_id)))

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=hyper_parameters['hidden_sizes'],
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = PyTorch_TRPO(env_spec=env.spec,
                        policy=policy,
                        value_function=value_function,
                        max_path_length=hyper_parameters['max_path_length'],
                        discount=hyper_parameters['discount'],
                        gae_lambda=hyper_parameters['gae_lambda'])

    runner.setup(algo, env)
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=hyper_parameters['batch_size'])
