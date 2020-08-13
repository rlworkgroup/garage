"""PyTorch algorithms."""
# isort:skip_file

# VPG has to be imported first because it is depended by A2C,
# PPO, and TRPO. PPO, TRPO, and VPG need to be imported
# before their MAML variants

from garage.torch.algos.vpg import VPG
from garage.torch.algos.a2c import A2C
from garage.torch.algos.bc import BC
from garage.torch.algos.ddpg import DDPG

from garage.torch.algos.maml_vpg import MAMLVPG
from garage.torch.algos.ppo import PPO
from garage.torch.algos.maml_ppo import MAMLPPO
from garage.torch.algos.trpo import TRPO
from garage.torch.algos.maml_trpo import MAMLTRPO
# SAC needs to be imported before MTSAC
from garage.torch.algos.sac import SAC
from garage.torch.algos.mtsac import MTSAC
from garage.torch.algos.pearl import PEARL

__all__ = [
    'A2C', 'BC', 'DDPG', 'VPG', 'PPO', 'TRPO', 'MAMLPPO', 'MAMLTRPO',
    'MAMLVPG', 'MTSAC', 'PEARL', 'SAC'
]
