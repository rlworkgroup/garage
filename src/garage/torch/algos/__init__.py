"""PyTorch algorithms."""
from garage.torch.algos._utils import _Default  # noqa: F401
from garage.torch.algos._utils import compute_advantages  # noqa: F401
from garage.torch.algos._utils import filter_valids  # noqa: F401
from garage.torch.algos._utils import make_optimizer  # noqa: F401
from garage.torch.algos._utils import pad_to_last  # noqa: F401
from garage.torch.algos.ddpg import DDPG
# VPG has to been import first because it is depended by MAML, PPO and TRPO.
from garage.torch.algos.vpg import VPG
from garage.torch.algos.maml import MAML  # noqa: I100
from garage.torch.algos.ppo import PPO
from garage.torch.algos.trpo import TRPO

__all__ = ['DDPG', 'VPG', 'PPO', 'TRPO', 'MAML']
