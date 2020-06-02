"""PyTorch-backed modules and algorithms."""
from garage.torch._utils import compute_advantages
from garage.torch._utils import filter_valids
from garage.torch._utils import pad_to_last

__all__ = ['compute_advantages', 'filter_valids', 'pad_to_last']
