#!/usr/bin/env python3
"""This is an example to train MAML-TRPO on ML10 environment."""
from metaworld.benchmarks import ML10
import torch

from garage import wrap_experiment
from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.envs.metaworld_task_names_wrapper import MetaworldTaskNamesWrapper
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy


@wrap_experiment(snapshot_mode='all')
def maml_trpo(ctxt=None, seed=1):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GarageEnv(
        normalize(MetaworldTaskNamesWrapper(ML10.get_train_tasks()),
                  expected_action_scale=10.))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 100),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    rollouts_per_task = 10
    max_path_length = 100

    runner = LocalRunner(ctxt)
    algo = MAMLTRPO(env=env,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=max_path_length,
                    meta_batch_size=20,
                    discount=0.99,
                    gae_lambda=1.,
                    inner_lr=0.1,
                    num_grad_updates=1)

    runner.setup(algo, env)
    runner.train(n_epochs=300, batch_size=rollouts_per_task * max_path_length)


maml_trpo(seed=1)
