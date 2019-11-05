"""PyTorch algorithms."""
from garage.torch.algos.ddpg import DDPG
from garage.torch.algos.vpg import VPG
from garage.torch.algos.ppo import PPO  # noqa: I100
from garage.torch.algos.trpo import TRPO

__all__ = ['DDPG', 'VPG', 'PPO', 'TRPO']
