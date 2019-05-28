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
    temp_dir = tempfile.TemporaryDirectory()
    snapshot_config = SnapshotConfig(
        snapshot_dir=temp_dir.name, snapshot_mode='all', snapshot_gap=1)

    @classmethod
    def teardown_class(cls):
        cls.temp_dir.cleanup()

    def test_before_load(self):
        fixture_exp(self.__class__.snapshot_config)

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
