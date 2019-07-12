from unittest import mock

import numpy as np
import pytest
import torch

from garage.torch.policies import GaussianMLPPolicy

test_settings = [
    (1, 1, (1, )),
    (1, 2, (2, )),
    (1, 3, (3, )),
    (1, 1, (1, 2)),
    (1, 2, (2, 1)),
    (1, 3, (4, 5)),
    (2, 1, (1, )),
    (2, 2, (2, )),
    (2, 3, (3, )),
    (2, 1, (1, 2)),
    (2, 2, (2, 1)),
    (2, 3, (4, 5)),
    (5, 1, (1, )),
    (5, 2, (2, )),
    (5, 3, (3, )),
    (5, 1, (1, 2)),
    (5, 2, (2, 1)),
    (5, 3, (4, 5)),
]


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', test_settings)
@mock.patch('garage.torch.modules.GaussianMLPModule')
def test_policy_get_actions(mock_model, input_dim, output_dim, hidden_sizes):
    action = torch.randn((output_dim, ))

    mock_dist = mock.MagicMock()
    mock_dist.rsample.return_value = action

    mock_model.return_value = mock_dist

    env_spec = mock.MagicMock()
    env_spec.observation_space.flat_dim = input_dim
    env_spec.action_space.flat_dim = output_dim

    policy = GaussianMLPPolicy(env_spec, mock_model)

    input = torch.ones(input_dim)
    sample = policy.get_actions(input)

    assert np.array_equal(sample, action.detach().numpy())


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', test_settings)
@mock.patch('garage.torch.modules.GaussianMLPModule')
def test_policy_get_action(mock_model, input_dim, output_dim, hidden_sizes):
    action = torch.randn((
        1,
        output_dim,
    ))

    mock_dist = mock.MagicMock()
    mock_dist.rsample.return_value = action

    mock_model.return_value = mock_dist

    env_spec = mock.MagicMock()
    env_spec.observation_space.flat_dim = input_dim
    env_spec.action_space.flat_dim = output_dim

    policy = GaussianMLPPolicy(env_spec, mock_model)

    input = torch.ones(input_dim)
    sample = policy.get_action(input)

    assert np.array_equal(sample, np.squeeze(action.detach().numpy()))
