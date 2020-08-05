#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm.

Here it runs MemorizeDigits-v0 environment with 1000 iterations.
"""
import click

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianCNNBaseline
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
        env = normalize(GymEnv('MemorizeDigits-v0', is_image=True))
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      filters=(
                                                  (32, (5, 5)),
                                                  (64, (3, 3)),
                                                  (64, (2, 2)),
                                              ),
                                      strides=(4, 2, 1),
                                      padding='VALID',
                                      hidden_sizes=(256, ))  # yapf: disable

        baseline = GaussianCNNBaseline(
            env_spec=env.spec,
            filters=(
                (32, (5, 5)),
                (64, (3, 3)),
                (64, (2, 2)),
            ),
            strides=(4, 2, 1),
            padding='VALID',
            hidden_sizes=(256, ),
            use_trust_region=True)  # yapf: disable

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   max_episode_length=100,
                   discount=0.99,
                   gae_lambda=0.95,
                   lr_clip_range=0.2,
                   policy_ent_coeff=0.0,
                   optimizer_args=dict(
                       batch_size=32,
                       max_episode_length=10,
                       learning_rate=1e-3,
                   ))

        runner.setup(algo, env)
        runner.train(n_epochs=1000, batch_size=batch_size)


ppo_memorize_digits()
