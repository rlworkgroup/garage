import pickle
import unittest

from garage.spaces.tuple import Tuple
from garage.tf.spaces import Discrete


class TestTuple(unittest.TestCase):
    def test_pickleable(self):
        obj = Tuple((Discrete(3), Discrete(2)))
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
        assert round_trip.components == obj.components
