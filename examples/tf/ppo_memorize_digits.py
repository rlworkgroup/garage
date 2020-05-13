#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm.

Here it runs MemorizeDigits-v0 environment with 1000 iterations.
"""
import click
import gym

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianCNNBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalCNNPolicy


@click.command()
@click.option('--batch_size', type=int, default=4000)
@wrap_experiment
def ppo_memorize_digits(ctxt=None, seed=1, batch_size=4000):
    """Train PPO on MemorizeDigits-v0 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        batch_size (int): Number of timesteps to use in each training step.

    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        env = TfEnv(normalize(gym.make('MemorizeDigits-v0')), is_image=True)
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      num_filters=(32, 64, 64),
                                      filter_dims=(5, 3, 2),
                                      strides=(4, 2, 1),
                                      padding='VALID',
                                      hidden_sizes=(256, ))

        baseline = GaussianCNNBaseline(env_spec=env.spec,
                                       regressor_args=dict(
                                           num_filters=(32, 64, 64),
                                           filter_dims=(5, 3, 2),
                                           strides=(4, 2, 1),
                                           padding='VALID',
                                           hidden_sizes=(256, ),
                                           use_trust_region=True))

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   max_path_length=100,
                   discount=0.99,
                   gae_lambda=0.95,
                   lr_clip_range=0.2,
                   policy_ent_coeff=0.0,
                   optimizer_args=dict(
                       batch_size=32,
                       max_epochs=10,
                       tf_optimizer_args=dict(learning_rate=1e-3),
                   ),
                   flatten_input=False)

        runner.setup(algo, env)
        runner.train(n_epochs=1000, batch_size=batch_size)


ppo_memorize_digits()
