import base64

import cloudpickle

from garage.experiment import SnapshotConfig
from garage.experiment.experiment_wrapper import run_experiment


def method_call(snapshot_config, variant_data):
    assert isinstance(snapshot_config, SnapshotConfig)
    assert snapshot_config.snapshot_dir == 'data/'
    assert snapshot_config.snapshot_mode == 'last'
    assert snapshot_config.snapshot_gap == 1
    assert variant_data is None


class TestExperimentWrapper:
    def test_experiment_wrapper_method_call(self):
        data = base64.b64encode(cloudpickle.dumps(method_call)).decode('utf-8')
        args = [
            '', '--args_data', data, '--use_cloudpickle', 'True', '--log_dir',
            'data/'
        ]
        run_experiment(args)
