"""
Auxiliary coverage test for logger.

This test covers snapshot and parameter logging.
"""
from types import SimpleNamespace
import unittest

from garage.logger import logger, snapshotter, StdOutput, tabular
from garage.misc.console import mkdir_p


class TestLogger(unittest.TestCase):
    def test_logger(self):
        args = SimpleNamespace()
        args.args_data = ""

        snap_dir = "exp"
        mkdir_p(snap_dir)

        snapshotter.snapshot_dir = snap_dir
        snapshotter.save_itr_params(1, {})

        logger.add_output(StdOutput())
        tabular.record_misc_stat("key", 1)

        logger.log(tabular)
