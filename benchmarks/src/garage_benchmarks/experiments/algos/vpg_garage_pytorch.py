"""A regression test for automatic benchmarking garage-PyTorch-VPG."""
import gym
import torch

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.torch.algos import VPG as PyTorch_VPG
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from garage.torch.value_functions import GaussianMLPValueFunction

hyper_parameters = {
    'hidden_sizes': [64, 64],
    'center_adv': True,
    'learning_rate': 1e-2,
    'discount': 0.99,
    'n_epochs': 250,
    'max_path_length': 100,
    'batch_size': 2048,
}


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

    env = GarageEnv(normalize(gym.make(env_id)))

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=hyper_parameters['hidden_sizes'],
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    policy_optimizer = OptimizerWrapper((torch.optim.Adam, dict(lr=2.5e-4)),
                                        policy,
                                        max_optimization_epochs=10,
                                        minibatch_size=64)
    vf_optimizer = OptimizerWrapper((torch.optim.Adam, dict(lr=2.5e-4)),
                                    value_function,
                                    max_optimization_epochs=10,
                                    minibatch_size=64)

    algo = PyTorch_VPG(env_spec=env.spec,
                       policy=policy,
                       value_function=value_function,
                       policy_optimizer=policy_optimizer,
                       vf_optimizer=vf_optimizer,
                       max_path_length=hyper_parameters['max_path_length'],
                       discount=hyper_parameters['discount'],
                       center_adv=hyper_parameters['center_adv'])

    runner.setup(algo, env)
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=hyper_parameters['batch_size'])
