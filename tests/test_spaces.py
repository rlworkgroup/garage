import gym
import numpy as np

from rllab.envs.gym_util.space_util import flatten, flatten_n, sample, unflatten, unflatten_n

from rllab.spaces import Box
from rllab.spaces import Discrete
from rllab.spaces import Product


def test_tuple_space():
    _ = gym.spaces.Tuple([gym.spaces.Discrete(3), gym.spaces.Discrete(2)])
    tuple_space = gym.spaces.Tuple(
        gym.spaces.Discrete(3), gym.spaces.Discrete(2))
    sample = sample(tuple_space)
    assert tuple_space.contains(sample)


def test_tuple_space_unflatten_n():
    space = gym.spaces.Tuple([gym.spaces.Discrete(3), gym.spaces.Discrete(3)])
    np.testing.assert_array_equal(
        flatten(space, (2, 2)),
        flatten_n(space, [(2, 2)])[0])
    np.testing.assert_array_equal(
        unflatten(space, flatten(space, (2, 2))),
        unflatten_n(space, flatten_n(space, [(2, 2)]))[0])


def test_box():
    space = gym.spaces.Box(low=-1, high=1, shape=(2, 2), dtype=np.float32)
    np.testing.assert_array_equal(
        flatten(space, [[1, 2], [3, 4]]), [1, 2, 3, 4])
    np.testing.assert_array_equal(
        flatten_n(space, [[[1, 2], [3, 4]]]), [[1, 2, 3, 4]])
    np.testing.assert_array_equal(
        unflatten(space, [1, 2, 3, 4]), [[1, 2], [3, 4]])
    np.testing.assert_array_equal(
        unflatten_n(space, [[1, 2, 3, 4]]), [[[1, 2], [3, 4]]])
