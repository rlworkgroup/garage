import numpy as np
import pytest

from garage.np.policies import FixedPolicy


def test_vectorization_multi_raises():
    policy = FixedPolicy(None, np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        policy.reset([True, True])
    with pytest.raises(ValueError):
        policy.get_actions(np.array([0, 0]))


def test_get_actions():
    policy = FixedPolicy(None, np.array([1, 2, 3]))
    assert policy.get_actions(np.array([0]).reshape(1, 1))[0] == 1
    assert policy.get_action(np.array([0]))[0] == 2
    assert policy.get_action(np.array([0]))[0] == 3
    with pytest.raises(IndexError):
        policy.get_action(np.ndarray([0]))
