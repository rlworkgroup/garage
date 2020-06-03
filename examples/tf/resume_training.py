#!/usr/bin/env python3
"""This is an example to resume training programmatically."""
# pylint: disable=no-value-for-parameter
import click

from garage import wrap_experiment
from garage.experiment import LocalTFRunner


@click.command()
@click.option('--saved_dir',
              required=True,
              help='Path where snapshots are saved.')
@wrap_experiment
def resume_experiment(ctxt, saved_dir):
    """Resume a Tensorflow experiment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        saved_dir (str): Path where snapshots are saved.

    """
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        runner.restore(from_dir=saved_dir)
        runner.resume()


resume_experiment()
