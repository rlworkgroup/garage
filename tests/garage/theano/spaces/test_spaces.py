import unittest

import numpy as np

from garage.theano.spaces import Box
from garage.theano.spaces import Discrete
from garage.theano.spaces import Tuple


class TestSpaces(unittest.TestCase):
    def test_tuple_space(self):
        _ = Tuple([Discrete(3), Discrete(2)])
        tuple_space = Tuple(Discrete(3), Discrete(2))
        sample = tuple_space.sample()
        assert tuple_space.contains(sample)

    def test_tuple_space_unflatten_n(self):
        space = Tuple([Discrete(3), Discrete(3)])
        np.testing.assert_array_equal(
            space.flatten((2, 2)),
            space.flatten_n([(2, 2)])[0])
        np.testing.assert_array_equal(
            space.unflatten(space.flatten((2, 2))),
            space.unflatten_n(space.flatten_n([(2, 2)]))[0])

    def test_box(self):
        space = Box(low=-1, high=1, shape=(2, 2))
        np.testing.assert_array_equal(
            space.flatten([[1, 2], [3, 4]]), [1, 2, 3, 4])
        np.testing.assert_array_equal(
            space.flatten_n([[[1, 2], [3, 4]]]), [[1, 2, 3, 4]])
        np.testing.assert_array_equal(
            space.unflatten([1, 2, 3, 4]), [[1, 2], [3, 4]])
        np.testing.assert_array_equal(
            space.unflatten_n([[1, 2, 3, 4]]), [[[1, 2], [3, 4]]])
