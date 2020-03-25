#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm (PyTorch).

Uses Ray sampler instead of OnPolicyVectorizedSampler.
Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import numpy as np
import ray
import torch

from garage import wrap_experiment
from garage.experiment import deterministic, LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.envs import TfEnv
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy


@wrap_experiment(snapshot_mode='none')
def trpo_ray_pendulum(ctxt=None, seed=1):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    # Since this is an example, we are running ray in a reduced state.
    # One can comment this line out in order to run ray at full capacity
    ray.init(memory=52428800,
             object_store_memory=78643200,
             ignore_reinit_error=True,
             log_to_driver=False,
             include_webui=False)
    deterministic.set_seed(seed)
    env = TfEnv(env_name='InvertedDoublePendulum-v2')

    runner = LocalRunner(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                center_adv=False)

    runner.setup(algo, env, sampler_cls=RaySampler)
    runner.train(n_epochs=100, batch_size=1024)


s = np.random.randint(0, 1000)
trpo_ray_pendulum(seed=s)
