#!/usr/bin/env python3
"""
This is an example to resume training programmatically.
"""
from garage.experiment import LocalRunner, run_experiment


def run_task(*_):
    with LocalRunner() as runner:
        runner.restore(from_dir='dir/', from_epoch=2)
        runner.resume()


run_experiment(
    run_task,
    log_dir='new_dir/',
    snapshot_mode='last',
    seed=1,
)
