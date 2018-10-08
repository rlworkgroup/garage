import pickle
import unittest

import gym
import numpy as np

from garage.tf.spaces.dict import Dict


class TestDict(unittest.TestCase):
    def test_pickleable(self):
        spaces = {
            "achieved_goal":
            gym.spaces.Box(
                low=-200., high=200., shape=(3, ), dtype=np.float32),
            "desired_goal":
            gym.spaces.Box(
                low=-200., high=200., shape=(3, ), dtype=np.float32),
            "observation":
            gym.spaces.Box(
                low=-200., high=200., shape=(25, ), dtype=np.float32)
        }
        obj = Dict(spaces)
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
        assert round_trip.contains(spaces)
