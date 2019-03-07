import unittest

import numpy as np

from garage.logger import logger


class TestTensorboard(unittest.TestCase):
    def test_histogram(self):
        vals = np.ones((2, 3, 4))
        logger.record_histogram('key', vals)
