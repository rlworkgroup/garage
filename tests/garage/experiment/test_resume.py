import tempfile

import numpy as np
import tensorflow as tf

from garage.experiment import SnapshotConfig
from garage.tf.experiment import LocalTFRunner
from tests.fixtures import TfGraphTestCase
from tests.fixtures.experiment import fixture_exp


class TestResume(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.snapshot_config = SnapshotConfig(snapshot_dir=self.temp_dir.name,
                                              snapshot_mode='last',
                                              snapshot_gap=1)
        self.policy_params = fixture_exp(self.snapshot_config, self.sess)
        for c in self.graph.collections:
            self.graph.clear_collection(c)

    def teardown_method(self):
        self.temp_dir.cleanup()
        super().teardown_method()

    def test_resume(self):
        sess = tf.compat.v1.Session(graph=tf.Graph())
        with LocalTFRunner(self.snapshot_config, sess) as runner:
            args = runner.restore(self.temp_dir.name)
            assert np.equal(
                runner.policy.get_param_values(),
                self.policy_params).all(), 'Policy parameters should persist'
            assert args.n_epochs == 5, (
                'Snapshot should save training parameters')
            assert args.start_epoch == 5, (
                'Last experiment should end at 5th iterations')

            batch_size = runner.train_args.batch_size
            n_epoch_cycles = runner.train_args.n_epoch_cycles

            runner.resume(n_epochs=10,
                          plot=False,
                          store_paths=True,
                          pause_for_plot=False)

            assert runner.train_args.n_epochs == 10
            assert runner.train_args.batch_size == batch_size
            assert runner.train_args.n_epoch_cycles == n_epoch_cycles
            assert not runner.train_args.plot
            assert runner.train_args.store_paths
            assert not runner.train_args.pause_for_plot
