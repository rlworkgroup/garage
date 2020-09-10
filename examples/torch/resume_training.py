#!/usr/bin/env python3
"""This is an example to resume training programmatically."""
# pylint: disable=no-value-for-parameter
import click

from garage import wrap_experiment
from garage.experiment import LocalRunner
from garage.torch import set_gpu_mode

import torch

@click.command()
@click.option('--saved_dir',
              required=True,
              help='Path where snapshots are saved.')
@wrap_experiment
def resume_experiment(ctxt, saved_dir):
    """Resume a PyTorch experiment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        saved_dir (str): Path where snapshots are saved.

    """

    if torch.cuda.is_available():
        set_gpu_mode(True)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        set_gpu_mode(False)
    runner = LocalRunner(snapshot_config=ctxt)
    runner.restore(from_dir=saved_dir)
    runner.resume()


resume_experiment()
