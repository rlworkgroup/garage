"""PyTorch algorithms."""
from garage.torch.algos.ddpg import DDPG
from garage.torch.algos.sac import SAC
from garage.torch.algos.vpg import VPG

__all__ = ['DDPG', 'VPG', 'SAC']
