#!/usr/bin/env python3
"""This is an example to train a task with CMA-ES.

Here it runs CartPole-v1 environment with 100 epoches.

Results:
    AverageReturn: 100
    RiseTime: epoch 38 (itr 760),
              but regression is observed in the course of training.
"""
from garage.experiment import run_experiment
from garage.np.algos import CMAES
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import OnPolicyVectorizedSampler
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalMLPPolicy


def run_task(snapshot_config, *_):
    """Train CMA_ES with Cartpole-v1 environment."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(env_name='CartPole-v1')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        n_samples = 20

        algo = CMAES(env_spec=env.spec,
                     policy=policy,
                     baseline=baseline,
                     max_path_length=100,
                     n_samples=n_samples)

        runner.setup(algo, env, sampler_cls=OnPolicyVectorizedSampler)
        # NOTE: make sure that n_epoch_cycles == n_samples !
        runner.train(n_epochs=100, batch_size=1000, n_epoch_cycles=n_samples)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
