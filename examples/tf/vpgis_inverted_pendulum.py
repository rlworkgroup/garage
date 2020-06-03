#!/usr/bin/env python3
"""Example using VPG with ISSampler.

Iterations alternate between live and importance sampled iterations.
"""
import gym

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import ISSampler
from garage.tf.algos import VPG
from garage.tf.policies import GaussianMLPPolicy


@wrap_experiment
def vpgis_inverted_pendulum(ctxt=None, seed=1):
    """Train TRPO with InvertedPendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        env = GarageEnv(normalize(gym.make('InvertedPendulum-v2')))

        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = VPG(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            max_kl_step=0.01,
        )

        runner.setup(algo,
                     env,
                     sampler_cls=ISSampler,
                     sampler_args=dict(n_backtrack=1))
        runner.train(n_epochs=40, batch_size=4000)


vpgis_inverted_pendulum()
