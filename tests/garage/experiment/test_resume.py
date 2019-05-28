import tempfile

import numpy as np

from garage.experiment import LocalRunner, SnapshotConfig
from tests.fixtures import TfGraphTestCase
from tests.fixtures.experiment import fixture_exp


class TestResume(TfGraphTestCase):
    temp_dir = tempfile.TemporaryDirectory()
    snapshot_config = SnapshotConfig(
        snapshot_dir=temp_dir.name, snapshot_mode='last', snapshot_gap=1)
    policy_params = None

    @classmethod
    def teardown_class(cls):
        cls.temp_dir.cleanup()

    def test_before_resume(self):
        self.__class__.policy_params = fixture_exp(
            self.__class__.snapshot_config)

    def test_resume(self):
        with LocalRunner(self.__class__.snapshot_config) as runner:
            args = runner.restore(self.__class__.temp_dir.name)
            assert np.isclose(
                runner.policy.get_param_values(), self.__class__.
                policy_params).all(), 'Policy parameters should persist'
            assert args.n_epochs == 5, (
                'Snapshot should save training parameters')
            assert args.start_epoch == 5, (
                'Last experiment should end at 5th iterations')

            batch_size = runner.train_args.batch_size
            n_epoch_cycles = runner.train_args.n_epoch_cycles

            runner.resume(
                n_epochs=10,
                plot=False,
                store_paths=True,
                pause_for_plot=False)

            assert runner.train_args.n_epochs == 10
            assert runner.train_args.batch_size == batch_size
            assert runner.train_args.n_epoch_cycles == n_epoch_cycles
            assert not runner.train_args.plot
            assert runner.train_args.store_paths
            assert not runner.train_args.pause_for_plot
