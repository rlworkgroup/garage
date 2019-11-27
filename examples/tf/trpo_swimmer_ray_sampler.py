#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.

Uses Ray sampler instead of on_policy vectorized
sampler.
Here it runs Swimmer-v2 environment with 40 iterations.
"""
import gym

from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.samplers import RaySamplerTF

seed = 100


def run_task(snapshot_config, *_):
    """Run task.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): Configuration
            values for snapshotting.
        *_ (object): Hyperparameters (unused).

    """
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('Swimmer-v2'))

        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=500,
                    discount=0.99,
                    max_kl_step=0.01)

        runner.setup(algo,
                     env,
                     sampler_cls=RaySamplerTF,
                     sampler_args={'seed': seed})
        runner.train(n_epochs=40, batch_size=4000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=seed,
)
