"""PyTorch-backed modules and algorithms."""
# yapf: disable
from garage.torch._functions import (compute_advantages,
                                     dict_np_to_torch,
                                     filter_valids,
                                     flatten_batch,
                                     flatten_to_single_vector,
                                     global_device,
                                     NonLinearity,
                                     np_to_torch,
                                     pad_to_last,
                                     product_of_gaussians,
                                     set_gpu_mode,
                                     torch_to_np,
                                     TransposeImage,
                                     update_module_params)

# yapf: enable
__all__ = [
    'compute_advantages', 'dict_np_to_torch', 'filter_valids', 'flatten_batch',
    'global_device', 'np_to_torch', 'pad_to_last', 'product_of_gaussians',
    'set_gpu_mode', 'torch_to_np', 'update_module_params', 'NonLinearity',
    'flatten_to_single_vector', 'TransposeImage'
]
