#!/usr/bin/env python3
"""This is an example to train a task with parallel sampling."""
import click

from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.samplers import BatchSampler


@click.command()
@click.option('--batch_size', type=int, default=4000)
@click.option('--max_path_length', type=int, default=100)
@wrap_experiment
def trpo_cartpole_batch_sampler(ctxt=None,
                                seed=1,
                                batch_size=4000,
                                max_path_length=100):
    """Train TRPO with CartPole-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        batch_size (int): Number of timesteps to use in each training step.
        max_path_length (int): Number of timesteps to truncate paths to.

    """
    set_seed(seed)
    n_envs = batch_size // max_path_length
    with LocalTFRunner(ctxt, max_cpus=n_envs) as runner:
        env = GarageEnv(env_name='CartPole-v1')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=max_path_length,
                    discount=0.99,
                    max_kl_step=0.01)

        runner.setup(algo=algo,
                     env=env,
                     sampler_cls=BatchSampler,
                     sampler_args={'n_envs': n_envs})

        runner.train(n_epochs=100, batch_size=4000, plot=False)


trpo_cartpole_batch_sampler()
