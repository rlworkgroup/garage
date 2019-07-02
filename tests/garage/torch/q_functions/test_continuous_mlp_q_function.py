import unittest
from unittest.mock import MagicMock

import akro
import numpy as np
import torch
from torch import nn

from garage.torch.modules import MLPModule
from garage.torch.q_functions import ContinuousMLPQFunction
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestContinuousMLPQFunction(unittest.TestCase):
    def test_output_1d(self):
        env_spec = MagicMock(return_value=DummyBoxEnv())
        env_spec.action_space = MagicMock(
            akro.Box(
                low=torch.FloatTensor([-5.0]),
                high=torch.FloatTensor([5.0]),
                dtype=np.float32))
        env_spec.observation_space = MagicMock(
            akro.Box(
                low=torch.FloatTensor([-1.0]),
                high=torch.FloatTensor([1.0]),
                dtype=np.float32))

        nn_module = MLPModule(
            input_dim=1,
            output_dim=1,
            hidden_nonlinearity=None,
            hidden_sizes=(1, ),
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_)

        q_function = ContinuousMLPQFunction(env_spec, nn_module)
        obs = torch.FloatTensor([1.0])
        assert q_function.get_qval(obs)

    def test_output_2d(self):
        env_spec = MagicMock(return_value=DummyBoxEnv())
        env_spec.action_space = MagicMock(
            akro.Box(
                low=torch.FloatTensor([-5.0, -5.0, -5.0]),
                high=torch.FloatTensor([5.0, 5.0, 5.0]),
                dtype=np.float32))
        env_spec.observation_space = MagicMock(
            akro.Box(
                low=torch.FloatTensor([-1.0, -1.0]),
                high=torch.FloatTensor([1.0, 1.0]),
                dtype=np.float32))

        nn_module = MLPModule(
            input_dim=2,
            output_dim=3,
            hidden_nonlinearity=None,
            hidden_sizes=(5, ),
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_)

        q_function = ContinuousMLPQFunction(env_spec, nn_module)
        obs = torch.FloatTensor([1.0, 1.0])
        expected_output = torch.FloatTensor([10.0, 10.0, 10.0])
        assert np.array_equal(
            torch.FloatTensor(q_function.get_qval(obs)), expected_output)
