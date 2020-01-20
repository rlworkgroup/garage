"""Collection of MAML algorithms."""
from garage.torch.algos.maml.maml_ppo import MAMLPPO
from garage.torch.algos.maml.maml_trpo import MAMLTRPO
from garage.torch.algos.maml.maml_vpg import MAMLVPG

__all__ = ['MAMLPPO', 'MAMLTRPO', 'MAMLVPG']
