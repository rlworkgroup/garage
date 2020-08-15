#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.

Uses Ray sampler instead of on_policy vectorized
sampler.
Here it runs Swimmer-v2 environment with 40 iterations.
"""
import ray

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy


@wrap_experiment
def trpo_swimmer_ray_sampler(ctxt=None, seed=1):
    """tf_trpo_swimmer.

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
             log_to_driver=False)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        set_seed(seed)
        env = GymEnv('Swimmer-v2')

        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_episode_length=500,
                    discount=0.99,
                    max_kl_step=0.01)

        runner.setup(algo,
                     env,
                     sampler_cls=RaySampler,
                     sampler_args={'seed': seed})
        runner.train(n_epochs=40, batch_size=4000)


trpo_swimmer_ray_sampler(seed=100)
