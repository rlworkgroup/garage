"""PyTorch algorithms."""
from garage.torch.algos.ddpg import DDPG
from garage.torch.algos.sac import SAC
from garage.torch.algos.vpg import VPG
from garage.torch.algos.ppo import PPO  # noqa: I100

__all__ = ['DDPG', 'VPG', 'PPO', 'SAC']
