"""PyTorch-backed modules and algorithms."""
# yapf: disable
from garage.torch._functions import (as_torch, as_torch_dict,
                                     compute_advantages, expand_var,
                                     filter_valids, flatten_batch,
                                     flatten_to_single_vector, global_device,
                                     NonLinearity, output_height_2d,
                                     output_width_2d, pad_to_last, prefer_gpu,
                                     product_of_gaussians, set_gpu_mode,
                                     soft_update_model, torch_to_np,
                                     update_module_params)

# yapf: enable
__all__ = [
    'compute_advantages', 'as_torch_dict', 'filter_valids', 'flatten_batch',
    'global_device', 'as_torch', 'pad_to_last', 'prefer_gpu',
    'product_of_gaussians', 'set_gpu_mode', 'soft_update_model', 'torch_to_np',
    'update_module_params', 'NonLinearity', 'flatten_to_single_vector',
    'output_width_2d', 'output_height_2d', 'expand_var'
]
