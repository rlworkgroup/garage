#!/usr/bin/env python3
"""This is an example to resume training programmatically."""
# pylint: disable=no-value-for-parameter
import click

from garage import wrap_experiment
from garage.trainer import Trainer


@click.command()
@click.option('--saved_dir',
              required=True,
              help='Path where snapshots are saved.')
@wrap_experiment
def resume_experiment(ctxt, saved_dir):
    """Resume a PyTorch experiment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        saved_dir (str): Path where snapshots are saved.

    """
    trainer = Trainer(snapshot_config=ctxt)
    trainer.restore(from_dir=saved_dir)
    trainer.resume()


resume_experiment()
