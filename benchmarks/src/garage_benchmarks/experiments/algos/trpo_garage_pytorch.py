"""A regression test for automatic benchmarking garage-PyTorch-TRPO."""
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.sampler import LocalSampler
from garage.torch.algos import TRPO as PyTorch_TRPO
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

hyper_parameters = {
    'hidden_sizes': [32, 32],
    'max_kl': 0.01,
    'gae_lambda': 0.97,
    'discount': 0.99,
    'n_epochs': 999,
    'batch_size': 1024,
}


@wrap_experiment
def trpo_garage_pytorch(ctxt, env_id, seed):
    """Create garage PyTorch TRPO model and training.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by Trainer to create the
                snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    trainer = Trainer(ctxt)

    env = normalize(GymEnv(env_id))

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=hyper_parameters['hidden_sizes'],
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length)

    algo = PyTorch_TRPO(env_spec=env.spec,
                        policy=policy,
                        value_function=value_function,
                        sampler=sampler,
                        discount=hyper_parameters['discount'],
                        gae_lambda=hyper_parameters['gae_lambda'])

    trainer.setup(algo, env)
    trainer.train(n_epochs=hyper_parameters['n_epochs'],
                  batch_size=hyper_parameters['batch_size'])
