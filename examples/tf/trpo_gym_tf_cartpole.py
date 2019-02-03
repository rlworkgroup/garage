#!/usr/bin/env python3

import gym

from garage.baselines import LinearFeatureBaseline
from garage.experiment import run_experiment
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy

# Need to wrap in a tf environment and force_reset to true
# see https://github.com/openai/rllab/issues/87#issuecomment-282519288
env = TfEnv(gym.make("CartPole-v0"))

policy = CategoricalMLPPolicy(
    name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=200,
    n_itr=120,
    discount=0.99,
    max_kl_step=0.01,
)

run_experiment(algo.train(), n_parallel=1, snapshot_mode="last", seed=1)
