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


class TestSnapshotter:

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def teardown_method(self):
        self.temp_dir.cleanup()

    @pytest.mark.parametrize('mode, files', [*configurations])
    def test_snapshotter(self, mode, files):
        snapshotter = Snapshotter(self.temp_dir.name, mode, 2)

        assert snapshotter.snapshot_dir == self.temp_dir.name
        assert snapshotter.snapshot_mode == mode
        assert snapshotter.snapshot_gap == 2

        snapshot_data = [{'testparam': 1}, {'testparam': 4}]
        snapshotter.save_itr_params(1, snapshot_data[0])
        snapshotter.save_itr_params(2, snapshot_data[1])

        for f, num in files.items():
            filename = osp.join(self.temp_dir.name, f)
            assert osp.exists(filename)
            with open(filename, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
                assert data == snapshot_data[num]

    def test_invalid_snapshot_mode(self):
        with pytest.raises(ValueError):
            snapshotter = Snapshotter(snapshot_dir=self.temp_dir.name,
                                      snapshot_mode='invalid')
            snapshotter.save_itr_params(2, {'testparam': 'invalid'})

    def test_get_available_itrs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            many, one, none = [
                tempfile.mkdtemp(dir=temp_dir) for _ in range(3)
            ]

            open(osp.join(many, 'itr_1.pkl'), 'a').close()
            open(osp.join(many, 'itr_3.pkl'), 'a').close()
            open(osp.join(many, 'itr_5.pkl'), 'a').close()
            assert Snapshotter.get_available_itrs(many) == [1, 3, 5]

            open(osp.join(one, 'params.pkl'), 'a').close()
            assert Snapshotter.get_available_itrs(one) == ['last']

            assert not Snapshotter.get_available_itrs(none)
