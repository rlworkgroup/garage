#!/usr/bin/env python3
"""This is an example to resume training programmatically."""
from garage.experiment import run_experiment
from garage.tf.experiment import LocalTFRunner


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        runner.restore(from_dir='dir/', from_epoch=2)
        runner.resume()


run_experiment(
    run_task,
    log_dir='new_dir/',
    snapshot_mode='last',
    seed=1,
)
