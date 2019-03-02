from os import path as osp
import pickle
import random
import shutil
import unittest

from garage.logger import snapshotter
from garage.misc.console import mkdir_p


class TestSnapshotter(unittest.TestCase):
    def test_snapshotter(self):
        snap_dir = "test_snapshotter"
        try:
            snapshotter.snapshot_dir = snap_dir
            assert snapshotter.snapshot_dir == snap_dir

            snapfiles = {
                'all': {
                    'itr_1.pkl': 0,
                    'itr_2.pkl': 1
                },
                'last': {
                    'params.pkl': 1
                },
                'gap': {
                    'itr_2.pkl': 1
                },
                'gap_and_last': {
                    'itr_2.pkl': 1,
                    'params.pkl': 1
                }
            }

            for mode, files in snapfiles.items():
                shutil.rmtree(snap_dir)
                mkdir_p(snap_dir)

                snapshotter.snapshot_mode = mode
                assert snapshotter.snapshot_mode == mode
                snapshotter.snapshot_gap = 2
                assert snapshotter.snapshot_gap == 2
                snapshot_data = []
                snapshot_data.append({'testparam': random.randint(0, 999)})
                snapshot_data.append({'testparam': random.randint(0, 999)})
                snapshotter.save_itr_params(1, snapshot_data[0])
                snapshotter.save_itr_params(2, snapshot_data[1])

                # check none/invalid modes
                snapshotter.snapshot_mode = 'none'
                snapshotter.save_itr_params(2, {'testparam': 'none'})
                snapshotter.snapshot_mode = 'invalid'
                try:
                    snapshotter.save_itr_params(2, {'testparam': 'invalid'})
                except TypeError as e:
                    assert str(e) == "Invalid snapshot mode invalid"
                except Exception as e:
                    self.fail("Wrong exception raised: " + str(e))
                else:
                    self.fail("Snapshot mode exception not raised.")

                for f, num in files.items():
                    filename = osp.join(snap_dir, f)
                    assert osp.exists(filename)
                    with open(filename, 'rb') as pkl_file:
                        data = pickle.load(pkl_file)
                        assert data == snapshot_data[num]
        finally:
            snapshotter.reset()
            shutil.rmtree(snap_dir)
