#!/usr/bin/env python3
"""
This is an example to train a task with PPO algorithm.

Here it runs MemorizeDigits-v0 environment with 1000 iterations.
"""
import gym

from garage.envs import normalize
from garage.experiment import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianCNNBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalCNNPolicy


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(normalize(gym.make('MemorizeDigits-v0')))
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      conv_filters=(32, 64, 64),
                                      conv_filter_sizes=(5, 3, 2),
                                      conv_strides=(4, 2, 1),
                                      conv_pad='VALID',
                                      hidden_sizes=(256, ))

        baseline = GaussianCNNBaseline(env_spec=env.spec,
                                       regressor_args=dict(
                                           conv_filters=(32, 64, 64),
                                           conv_filter_sizes=(5, 3, 2),
                                           conv_strides=(4, 2, 1),
                                           conv_pads=('VALID', 'VALID'),
                                           hidden_sizes=(256, ),
                                           use_trust_region=True))

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   max_path_length=100,
                   discount=0.99,
                   max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=1000, batch_size=2048)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
