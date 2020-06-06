"""PyTorch algorithms."""
from garage.torch.algos.ddpg import DDPG
# VPG has to been import first because it is depended by PPO and TRPO.
from garage.torch.algos.vpg import VPG
from garage.torch.algos.ppo import PPO  # noqa: I100
from garage.torch.algos.trpo import TRPO
from garage.torch.algos.maml_ppo import MAMLPPO  # noqa: I100
from garage.torch.algos.maml_trpo import MAMLTRPO
from garage.torch.algos.maml_vpg import MAMLVPG
from garage.torch.algos.pearl import PEARL
from garage.torch.algos.sac import SAC
from garage.torch.algos.mtsac import MTSAC  # noqa: I100

__all__ = [
    'DDPG', 'VPG', 'PPO', 'TRPO', 'MAMLPPO', 'MAMLTRPO', 'MAMLVPG', 'MTSAC',
    'PEARL', 'SAC'
]
