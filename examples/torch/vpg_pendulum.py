#!/usr/bin/env python3
"""This is an example to train a task with VPG algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.

Results:
    AverageReturn: 450 - 650
"""
import torch

from garage.experiment import LocalRunner, run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.envs import TfEnv
from garage.torch.algos import VPG
from garage.torch.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Run the job."""
    env = TfEnv(env_name='InvertedDoublePendulum-v2')

    runner = LocalRunner(snapshot_config)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = VPG(env_spec=env.spec,
               policy=policy,
               optimizer=torch.optim.Adam,
               baseline=baseline,
               max_path_length=100,
               discount=0.99,
               center_adv=False,
               policy_lr=1e-2)

    runner.setup(algo, env)
    runner.train(n_epochs=100, batch_size=10000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
