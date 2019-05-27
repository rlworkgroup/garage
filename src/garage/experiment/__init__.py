"""Experiment functions."""
from garage.experiment.experiment import run_experiment
from garage.experiment.experiment import to_local_command
from garage.experiment.experiment import variant
from garage.experiment.experiment import VariantGenerator
from garage.experiment.snapshotter import Snapshotter

# LocalRunner needs snapshotter to be imported, so we have to use a strange
# import order here
snapshotter = Snapshotter()

from garage.experiment.local_tf_runner import LocalRunner  # noqa: I100,E402,E501,I202 pylint: disable=wrong-import-position

__all__ = [
    'run_experiment', 'to_local_command', 'variant', 'VariantGenerator',
    'LocalRunner', 'Snapshotter', 'snapshotter'
]
