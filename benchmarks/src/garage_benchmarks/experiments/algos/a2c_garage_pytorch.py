"""A regression test for automatic benchmarking garage-PyTorch-A2C."""
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.torch.algos.a2c_2 import A2C
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from garage.torch.value_functions import GaussianMLPValueFunction

hyper_parameters = {
    'hidden_sizes': [32, 32],
    'learning_rate': 1e-3,
    'discount': 0.99,
    'n_epochs': 1000,
    'batch_size': 2000,
}


@wrap_experiment
def a2c_garage_pytorch(ctxt, env_id, seed):
    """Create garage PyTorch A2C model and training.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    runner = LocalRunner(ctxt)

    env = normalize(GymEnv(env_id))

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=hyper_parameters['hidden_sizes'],
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                                hidden_sizes=(32, 32),
                                                hidden_nonlinearity=torch.tanh,
                                                output_nonlinearity=None)

    policy_optimizer = OptimizerWrapper(
        (torch.optim.Adam, dict(lr=hyper_parameters['learning_rate'])), policy)

    vf_optimizer = OptimizerWrapper(
        (torch.optim.Adam, dict(lr=hyper_parameters['learning_rate'])),
        value_function)

    algo = A2C(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        policy_optimizer=policy_optimizer,
        vf_optimizer=vf_optimizer,
        stop_entropy_gradient=True,
        center_adv=False,
        max_episode_length=env.spec.max_episode_length,
        discount=hyper_parameters['discount'],
    )

    runner.setup(algo, env)
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=hyper_parameters['batch_size'])
