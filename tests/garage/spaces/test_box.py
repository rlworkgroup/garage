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

    def test_same_dtype(self):
        type1 = np.float32
        box = Box(low=0, high=255, shape=(3, 4), dtype=type1)
        assert box.dtype == type1

        type2 = np.uint8
        box = Box(low=0, high=255, shape=(3, 4), dtype=type2)
        assert box.dtype == type2

    def test_invalid_env(self):
        with self.assertRaises(AttributeError):
            Box(low=0.0, high=1.0)

        with self.assertRaises(AssertionError):
            Box(low=np.array([-1.0, -2.0]),
                high=np.array([1.0, 2.0]),
                shape=(2, 2))

    def test_default_float32_env(self):
        box = Box(low=0.0, high=1.0, shape=(3, 4))
        assert box.dtype == np.float32

        box = Box(low=np.array([-1.0, -2.0]), high=np.array([1.0, 2.0]))
        assert box.dtype == np.float32

    def test_uint8_warning_env(self):
        with self.assertWarns(UserWarning):
            box = Box(low=0, high=255, shape=(3, 4))
            assert box.dtype == np.float32

        with self.assertWarns(UserWarning):
            box = Box(low=np.array([0, 0]), high=np.array([255, 255]))
            assert box.dtype == np.float32
