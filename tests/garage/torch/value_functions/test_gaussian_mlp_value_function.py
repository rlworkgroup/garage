import torch

from garage.envs import GymEnv
from garage.torch.value_functions import GaussianMLPValueFunction

from tests.fixtures.envs.dummy import DummyBoxEnv


def test_loss_shape_is_correct():
    box_env = GymEnv(DummyBoxEnv())
    vf = GaussianMLPValueFunction(box_env.spec)
    observation, _ = box_env.reset()
    obs = torch.as_tensor(observation).repeat(1000, 1)
    returns = torch.zeros(1000)
    assert vf.compute_loss(obs, returns).shape == ()


def test_forward_shape():
    box_env = GymEnv(DummyBoxEnv())
    vf = GaussianMLPValueFunction(box_env.spec)
    observation, _ = box_env.reset()
    obs = torch.as_tensor(observation).repeat(1000, 100, 1)
    assert vf.forward(obs).shape == (1000, 100)
