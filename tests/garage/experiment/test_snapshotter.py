from os import path as osp
import pickle
import tempfile

import pytest

from garage.experiment import Snapshotter

configurations = [('all', {
    'itr_1.pkl': 0,
    'itr_2.pkl': 1
}), ('last', {
    'params.pkl': 1
}), ('gap', {
    'itr_2.pkl': 1
}), ('gap_and_last', {
    'itr_2.pkl': 1,
    'params.pkl': 1
}), ('none', {})]


class TestSanpshotter:
    def setup_method(self):
        self.snapshot_dir = tempfile.TemporaryDirectory()
        self.snapshotter = Snapshotter()

    def teardown_method(self):
        self.snapshotter.reset()
        self.snapshot_dir.cleanup()

    def test_set_snapshot_dir(self):
        self.snapshotter.snapshot_dir = self.snapshot_dir.name
        assert self.snapshotter.snapshot_dir == self.snapshot_dir.name

    @pytest.mark.parametrize('mode, files', [*configurations])
    def test_snapshotter(self, mode, files):
        self.snapshotter.snapshot_dir = self.snapshot_dir.name

        self.snapshotter.snapshot_mode = mode
        assert self.snapshotter.snapshot_mode == mode
        self.snapshotter.snapshot_gap = 2
        assert self.snapshotter.snapshot_gap == 2

        snapshot_data = [{'testparam': 1}, {'testparam': 4}]
        self.snapshotter.save_itr_params(1, snapshot_data[0])
        self.snapshotter.save_itr_params(2, snapshot_data[1])

        for f, num in files.items():
            filename = osp.join(self.snapshot_dir.name, f)
            assert osp.exists(filename)
            with open(filename, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
                assert data == snapshot_data[num]

    def test_invalid_snapshot_mode(self):
        with pytest.raises(ValueError):
            self.snapshotter.snapshot_dir = self.snapshot_dir.name
            self.snapshotter.snapshot_mode = 'invalid'
            self.snapshotter.save_itr_params(2, {'testparam': 'invalid'})
