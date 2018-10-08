import pickle
import unittest

from garage.spaces.product import Product
from garage.theano.spaces import Discrete


class TestProduct(unittest.TestCase):
    def test_pickleable(self):
        obj = Product((Discrete(3), Discrete(2)))
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
        assert round_trip.components == obj.components
