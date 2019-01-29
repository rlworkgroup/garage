#!/usr/bin/env python3
"""
This example demonstrates how to use run_experiment to send a training job to
an Amazon EC2 cluster.

Here it trains the CartPole-v1 environment for 100 iterations on each of 3 step
sizes and 5 seeds.
"""
import sys

from garage.baselines import LinearFeatureBaseline
from garage.experiment import run_experiment
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy


def run_task(v):
    """
    We wrap the main training loop in the run_task function so that
    run_experiment can easily execute variants of the experiment on different
    machines
    """
    env = TfEnv(env_name="CartPole-v1")

    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers,
        # each with 32 hidden units.
        hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=40,
        discount=0.99,
        step_size=v["step_size"],
        # Uncomment both lines (this and the plot parameter below) to enable
        # plotting
        plot=True,
    )
    algo.train()


for step_size in [0.01, 0.05, 0.1]:
    for seed in [1, 11, 21, 31, 41]:
        run_experiment(
            run_task,
            exp_prefix="first_exp",
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a
            # random seed will be used
            seed=seed,
            # mode="local",
            mode="ec2",
            variant=dict(step_size=step_size, seed=seed)
            # plot=True,
            # terminate_machine=False,
        )
        sys.exit()
