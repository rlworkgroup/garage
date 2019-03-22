"""Allows the taking of snapshots."""
from os import path as osp
import pickle


class Snapshotter:
    """This class handles the creation of snapshots."""

    def __init__(self):
        self._snapshot_dir = None
        self._snapshot_mode = 'all'
        self._snapshot_gap = 1

    @property
    def snapshot_dir(self, ):
        """The directory to save the snapshots in."""
        return self._snapshot_dir

    @snapshot_dir.setter
    def snapshot_dir(self, dir_name):
        self._snapshot_dir = dir_name

    @property
    def snapshot_mode(self, ):
        """The type of snapshot to take.

        See save_itr_params.
        """
        return self._snapshot_mode

    @snapshot_mode.setter
    def snapshot_mode(self, mode):
        self._snapshot_mode = mode

    @property
    def snapshot_gap(self, ):
        """The number of iterations between snapshots."""
        return self._snapshot_gap

    @snapshot_gap.setter
    def snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def save_itr_params(self, itr, params):
        """Save the parameters if at the right iteration."""
        if self._snapshot_dir:
            if self._snapshot_mode == 'all':
                file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == "gap":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir,
                                         'itr_%d.pkl' % itr)
                    pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == "gap_and_last":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir,
                                         'itr_%d.pkl' % itr)
                    pickle.dump(params, open(file_name, "wb"))
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == 'none':
                pass
            else:
                raise TypeError(f'Invalid snapshot mode {self._snapshot_mode}')
