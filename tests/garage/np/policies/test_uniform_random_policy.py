import numpy as np
import pytest

from garage.envs import GymEnv, normalize
from garage.np.policies import UniformRandomPolicy


@pytest.mark.mujoco
def test_get_actions():
    env = normalize(GymEnv('InvertedDoublePendulum-v2'))
    policy = UniformRandomPolicy(env.spec)
    assert policy.get_actions(np.array([0]).reshape(1, 1))[0]
    assert policy.get_action(np.array([0]))[0]
