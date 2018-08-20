import pickle
import unittest

from garage.spaces.dict import Dict
from garage.spaces.discrete import Discrete


class TestDict(unittest.TestCase):
    def test_pickleable(self):
        obj = Dict({"position": Discrete(2), "velocity": Discrete(3)})
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
