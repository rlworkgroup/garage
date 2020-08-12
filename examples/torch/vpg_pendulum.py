#!/usr/bin/env python3
"""This is an example to train a task with VPG algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.

Results:
    AverageReturn: 450 - 650
"""
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.torch.algos import VPG
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction


@wrap_experiment
def vpg_pendulum(ctxt=None, seed=1):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GymEnv('InvertedDoublePendulum-v2')

    runner = LocalRunner(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = VPG(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               max_episode_length=100,
               discount=0.99,
               center_adv=False)

    runner.setup(algo, env)
    runner.train(n_epochs=100, batch_size=10000)


vpg_pendulum()
