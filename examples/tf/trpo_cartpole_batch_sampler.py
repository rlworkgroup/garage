#!/usr/bin/env python3
"""
This is an example to train a task with VPG algorithm.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
from garage.baselines import LinearFeatureBaseline
from garage.experiment.local_tf_runner import LocalRunner
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.samplers import BatchSampler

batch_size = 4000
max_path_length = 500
n_envs = batch_size // max_path_length

# Use BatchSampler.initialize() to setup workers before TF initialization.
# Note: Multiprocesses MUST be forked before TF initialization.
# Otherwise it will hang, because TF is not fork-safe.
BatchSampler.initialize(n_envs)

with LocalRunner() as runner:
    env = TfEnv(env_name="CartPole-v1")

    policy = CategoricalMLPPolicy(
        name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=max_path_length,
        discount=0.99,
        max_kl_step=0.01)

    runner.setup(
        algo=algo,
        env=env,
        sampler_cls=BatchSampler,
        sampler_args={'n_envs': n_envs})

    runner.train(n_epochs=100, batch_size=4000, plot=False)
