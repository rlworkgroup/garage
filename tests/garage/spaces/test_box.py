import pickle
import unittest

import numpy as np

from garage.spaces.box import Box


class TestBox(unittest.TestCase):
    def test_pickleable(self):
        obj = Box(-1.0, 1.0, (3, 4))
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
        assert round_trip.shape == obj.shape
        assert np.array_equal(round_trip.bounds[0], obj.bounds[0])
        assert np.array_equal(round_trip.bounds[1], obj.bounds[1])
