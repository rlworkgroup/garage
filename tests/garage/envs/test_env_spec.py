import pickle

import akro

from garage.envs.env_spec import EnvSpec


class TestEnvSpec:

    def test_pickleable(self):
        env_spec = EnvSpec(akro.Box(-1, 1, (1, )), akro.Box(-2, 2, (2, )), 500)
        round_trip = pickle.loads(pickle.dumps(env_spec))
        assert round_trip
        assert round_trip.action_space == env_spec.action_space
        assert round_trip.observation_space == env_spec.observation_space
        assert round_trip.max_episode_length == env_spec.max_episode_length
