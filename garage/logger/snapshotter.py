"""Allows the taking of snapshots."""
import errno
import os
from os import listdir
from os import path as osp
import pickle
import re

import joblib

from garage.misc.console import mkdir_p


class Snapshotter:
    """This class handles the creation of snapshots."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the snapshotter to default settings."""
        self._snapshot_dir = None
        self._snapshot_mode = 'all'
        self._snapshot_gap = 1

    @property
    def snapshot_dir(self, ):
        """Save snapshots in this directory."""
        return self._snapshot_dir

    @snapshot_dir.setter
    def snapshot_dir(self, dir_name):
        mkdir_p(dir_name)
        self._snapshot_dir = dir_name

    @property
    def snapshot_mode(self, ):
        """Take this type of snapshot.

        See save_itr_params.
        """
        return self._snapshot_mode

    @snapshot_mode.setter
    def snapshot_mode(self, mode):
        self._snapshot_mode = mode

    @property
    def snapshot_gap(self, ):
        """Wait this number of iterations before taking another snapshot."""
        return self._snapshot_gap

    @snapshot_gap.setter
    def snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def save_itr_params(self, itr, params):
        """Save the parameters if at the right iteration."""
        if self._snapshot_dir:
            file_name = None

            if self._snapshot_mode == 'all':
                file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
            elif self._snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
            elif self._snapshot_mode == 'gap':
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir,
                                         'itr_%d.pkl' % itr)
            elif self._snapshot_mode == 'gap_and_last':
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir,
                                         'itr_%d.pkl' % itr)
                file_name_last = osp.join(self._snapshot_dir, 'params.pkl')
                with open(file_name_last, 'wb') as file:
                    pickle.dump(params, file)
            elif self._snapshot_mode == 'none':
                pass
            else:
                raise ValueError('Invalid snapshot mode {}'.format(
                    self._snapshot_mode))

            if file_name:
                with open(file_name, 'wb') as file:
                    pickle.dump(params, file)

    def load(self, itr='last'):
        r"""Load one snapshot of parameters from disk.

        Args:
            itr(int or string): Iteration to load.
                Can be an integer, 'last' or 'first'.

        Returns:
            dict: Loaded snapshot

        """
        if isinstance(itr, int) or itr.isdigit():
            snapshot_file = osp.join(self._snapshot_dir,
                                     'itr_{}.pkl'.format(itr))
        else:
            if itr not in ('last', 'first'):
                raise ValueError(
                    "itr should be an integer or 'last' or 'first'")

            snapshot_file = osp.join(self._snapshot_dir, 'params.pkl')
            if not osp.isfile(snapshot_file):
                files = [
                    f for f in listdir(self._snapshot_dir)
                    if f.endswith('.pkl')
                ]
                if not files:
                    raise FileNotFoundError(errno.ENOENT,
                                            os.strerror(errno.ENOENT),
                                            '*.pkl file')
                files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))
                snapshot_file = files[0] if itr == 'first' else files[-1]
                snapshot_file = osp.join(self._snapshot_dir, snapshot_file)

        assert osp.isfile(snapshot_file)

        with open(snapshot_file, 'rb') as file:
            return joblib.load(file)
