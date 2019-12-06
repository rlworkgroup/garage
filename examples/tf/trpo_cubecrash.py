#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.

Here it runs CubeCrash-v0 environment with 100 iterations.
"""
import click
import gym

from garage.envs import normalize
from garage.experiment import run_experiment
from garage.tf.algos import TRPO
from garage.tf.baselines import GaussianCNNBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalCNNPolicy


def run_task(snapshot_config, variant_data, *_):
    """Run task.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.

        variant_data (dict): Custom arguments for the task.

        *_ (object): Ignored by this function.

    """
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(normalize(gym.make('CubeCrash-v0')))
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      conv_filters=(32, 64),
                                      conv_filter_sizes=(8, 4),
                                      conv_strides=(4, 2),
                                      conv_pad='VALID',
                                      hidden_sizes=(32, 32))

        baseline = GaussianCNNBaseline(env_spec=env.spec,
                                       regressor_args=dict(
                                           num_filters=(32, 64),
                                           filter_dims=(8, 4),
                                           strides=(4, 2),
                                           padding='VALID',
                                           hidden_sizes=(32, 32),
                                           use_trust_region=True))

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=100,
                    discount=0.99,
                    max_kl_step=0.01,
                    flatten_input=False)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=variant_data['batch_size'])


@click.command()
@click.option('--batch_size', '_batch_size', type=int, default=4000)
def _args(_batch_size):
    """A click command to parse arguments for automated testing purposes.

    Args:
        _batch_size (int): Number of environment steps in one batch.

    Returns:
        int: The input argument as-is.

    """
    return _batch_size


batch_size = _args.main(standalone_mode=False)
run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
    variant={'batch_size': batch_size},
)
