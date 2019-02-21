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
            mkdir_p(snap_dir)
            snapshotter.snapshot_dir = snap_dir

            snapmodes = ['all', 'last', 'gap', 'gap_and_last']
            snapfiles = [['itr_1.pkl'], ['params.pkl'], ['itr_1.pkl'],
                         ['params.pkl', 'itr_1.pkl']]

            for mode, files in zip(snapmodes, snapfiles):
                snapshotter.snapshot_mode = mode
                snapshot_data = {'testparam': random.randint(0, 999)}
                snapshotter.save_itr_params(1, snapshot_data)

                for f in files:
                    filename = osp.join(snap_dir, f)
                    assert osp.exists(filename)
                    with open(filename, 'rb') as pkl_file:
                        data = pickle.load(pkl_file)
                        assert data == snapshot_data
        finally:
            shutil.rmtree(snap_dir)
