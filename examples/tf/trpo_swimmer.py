#!/usr/bin/env python3

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.mujoco import SwimmerEnv
from garage.runners import LocalRunner
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy

with LocalRunner() as runner:
    env = TfEnv(
        normalize(
            SwimmerEnv(),
            normalize_obs=True,
            normalize_reward=True,
        ))

    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=500,
        discount=0.99,
        step_size=0.01)

    runner.setup(algo, env)
    runner.train(n_epochs=40, batch_size=4000)
