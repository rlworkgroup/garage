import tempfile

import pytest

from garage.experiment import SnapshotConfig, Snapshotter
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.experiment import fixture_exp

configurations = [('last', 4), ('first', 0), (3, 3)]


class TestSnapshot(TfGraphTestCase):
    def setup_method(self):
        super().setup_method()
        self.temp_dir = tempfile.TemporaryDirectory()
        snapshot_config = SnapshotConfig(
            snapshot_dir=self.temp_dir.name,
            snapshot_mode='all',
            snapshot_gap=1)
        fixture_exp(snapshot_config, self.sess)
        for c in self.graph.collections:
            self.graph.clear_collection(c)

    def teardown_method(self):
        self.temp_dir.cleanup()
        super().teardown_method()

    @pytest.mark.parametrize('load_mode, last_epoch', [*configurations])
    def test_load(self, load_mode, last_epoch):
        snapshotter = Snapshotter()
        saved = snapshotter.load(self.temp_dir.name, load_mode)

        assert isinstance(saved['algo'], VPG)
        assert isinstance(saved['env'], TfEnv)
        assert isinstance(saved['algo'].policy, CategoricalMLPPolicy)
        assert saved['last_epoch'] == last_epoch

    def test_load_with_invalid_load_mode(self):
        snapshotter = Snapshotter()
        with pytest.raises(ValueError):
            snapshotter.load(self.temp_dir.name, 'foo')
