#!/usr/bin/env python3

import gym

from garage.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.runners import LocalRunner

with LocalRunner() as runner:
    env = TfEnv(gym.make("CartPole-v0"))

    policy = CategoricalMLPPolicy(
        name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=200,
        discount=0.99,
        max_kl_step=0.01,
    )

    runner.setup(algo, env)
    runner.train(n_epochs=120, batch_size=4000)
