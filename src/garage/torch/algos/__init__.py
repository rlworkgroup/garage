"""PyTorch algorithms."""
from garage.torch.algos._utils import _Default  # noqa: F401
from garage.torch.algos._utils import compute_advantages  # noqa: F401
from garage.torch.algos._utils import filter_valids  # noqa: F401
from garage.torch.algos._utils import make_optimizer  # noqa: F401
from garage.torch.algos._utils import pad_to_last  # noqa: F401
from garage.torch.algos.ddpg import DDPG
from garage.torch.algos.vpg import VPG
from garage.torch.algos.ppo import PPO  # noqa: I100
from garage.torch.algos.sac import SAC
from garage.torch.algos.trpo import TRPO

__all__ = ['DDPG', 'VPG', 'PPO', 'SAC', 'TRPO']
