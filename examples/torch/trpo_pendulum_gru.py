#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy, GaussianGRUPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer


@wrap_experiment
def trpo_pendulum_gru(ctxt=None, seed=1):
    """Train TRPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GymEnv('InvertedDoublePendulum-v2')

    trainer = Trainer(ctxt)

    policy = GaussianGRUPolicy(env.spec,
                               hidden_dim=32, # [32, 32]
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)


    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                # max_path_length=100,
                discount=0.99,
                center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=100, batch_size=1024)


trpo_pendulum_gru(seed=1)
