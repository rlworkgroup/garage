#!/usr/bin/env python3
"""
This is an example to train a task with Cross Entropy Method.

Here it runs CartPole-v1 environment with 100 epoches.

Results:
    AverageReturn: 100
    RiseTime: epoch 8
"""
from garage.algos import CEM
from garage.baselines import LinearFeatureBaseline
from garage.experiment import LocalRunner, run_experiment
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.samplers import OnPolicyVectorizedSampler


def run_task(*_):
    """Train CEM with Cartpole-v1 environment."""
    with LocalRunner() as runner:
        env = TfEnv(env_name="CartPole-v1")

        policy = CategoricalMLPPolicy(
            name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        runner.initialize_tf_vars()

        n_samples = 20

        algo = CEM(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            best_frac=0.05,
            max_path_length=100,
            n_samples=n_samples)

        runner.setup(algo, env, sampler_cls=OnPolicyVectorizedSampler)
        # NOTE: make sure that n_epoch_cycles == n_samples !
        runner.train(n_epochs=100, batch_size=1000, n_epoch_cycles=n_samples)


run_experiment(
    run_task,
    snapshot_mode="last",
    seed=1,
)
