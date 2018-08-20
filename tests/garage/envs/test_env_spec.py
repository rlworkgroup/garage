import pickle
import unittest

from garage.envs.env_spec import EnvSpec
from garage.spaces import Box


class TestEnvSpec(unittest.TestCase):
    def test_pickleable(self):
        env_spec = EnvSpec(Box(-1, 1, (1)), Box(-2, 2, (2)))
        round_trip = pickle.loads(pickle.dumps(env_spec))
        assert round_trip
        assert round_trip.action_space == env_spec.action_space
        assert round_trip.observation_space == env_spec.observation_space
