import pickle
import unittest

from garage.spaces.box import Box


class TestBox(unittest.TestCase):
    def test_pickleable(self):
        obj = Box(-1.0, 1.0, (3, 4))
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
