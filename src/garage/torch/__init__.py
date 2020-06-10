"""PyTorch-backed modules and algorithms."""
from garage.torch._functions import compute_advantages
from garage.torch._functions import dict_np_to_torch
from garage.torch._functions import filter_valids
from garage.torch._functions import flatten_batch
from garage.torch._functions import global_device
from garage.torch._functions import pad_to_last
from garage.torch._functions import product_of_gaussians
from garage.torch._functions import set_gpu_mode
from garage.torch._functions import torch_to_np
from garage.torch._functions import update_module_params

__all__ = [
    'compute_advantages', 'dict_np_to_torch', 'filter_valids', 'flatten_batch',
    'global_device', 'pad_to_last', 'product_of_gaussians', 'set_gpu_mode',
    'torch_to_np', 'update_module_params'
]
