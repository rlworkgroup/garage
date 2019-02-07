"""
Auxiliary coverage test for logger.

This test covers snapshot and parameter logging.
"""
import os
from types import SimpleNamespace
import unittest

import matplotlib.pyplot as plt
import numpy as np

from garage.misc import logger


class TestLogger(unittest.TestCase):
    def test_logger(self):
        args = SimpleNamespace()
        args.args_data = ""

        logger.set_snapshot_dir("exp")
        logger.save_itr_params(1, {})
        logger.log_parameters_lite("exp-log", args)
        logger.log_variant("exp-log", {})
        logger.record_tabular_misc_stat("key", 1)
        if os.path.isfile("exp-log"):
            os.remove("exp-log")

    def test_record_matplotlib(self):
        fig = plt.figure()
        ax = fig.gca()
        xs = np.arange(10.0)
        ys = np.random.rand(10)
        ax.scatter(xs, ys)
        logger.record_matplotlib("foo", fig)
