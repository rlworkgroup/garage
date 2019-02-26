#!/usr/bin/env python3
"""
This is an example to train a task with VPG algorithm.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 16
"""
from garage.baselines import LinearFeatureBaseline
from garage.experiment import LocalRunner
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy

with LocalRunner() as runner:
    env = TfEnv(env_name='CartPole-v1')

    policy = CategoricalMLPPolicy(
        name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=100,
        discount=0.99,
        optimizer_args=dict(tf_optimizer_args=dict(learning_rate=0.01, )))

    runner.setup(algo, env)
    runner.train(n_epochs=100, batch_size=10000)
