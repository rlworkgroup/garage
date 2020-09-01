"""Reinforcement Learning Algorithms which use NumPy as a numerical backend."""
# yapf: disable
from garage.np._functions import (concat_tensor_dict_list,
                                  discount_cumsum,
                                  explained_variance_1d,
                                  flatten_tensors,
                                  pad_tensor,
                                  pad_tensor_dict,
                                  pad_tensor_n,
                                  paths_to_tensors,
                                  rrse,
                                  slice_nested_dict,
                                  sliding_window,
                                  stack_and_pad_tensor_dict_list,
                                  stack_tensor_dict_list,
                                  truncate_tensor_dict,
                                  unflatten_tensors)

# yapf: enable

__all__ = [
    'paths_to_tensors',
    'discount_cumsum',
    'explained_variance_1d',
    'flatten_tensors',
    'unflatten_tensors',
    'pad_tensor',
    'pad_tensor_n',
    'pad_tensor_dict',
    'stack_tensor_dict_list',
    'stack_and_pad_tensor_dict_list',
    'concat_tensor_dict_list',
    'truncate_tensor_dict',
    'slice_nested_dict',
    'rrse',
    'sliding_window',
]
