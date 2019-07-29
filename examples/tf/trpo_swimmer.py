#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
import gym

from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Run task."""
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

        runner.setup(algo, env)
        runner.train(n_epochs=40, batch_size=4000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
