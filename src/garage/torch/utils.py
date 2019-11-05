"""Utility functions for PyTorch."""
import torch


def np_to_torch(array_dict):
    """Convert numpy arrays to PyTorch tensors.

    Args:
        array_dict (dict): Dictionary of data in numpy arrays.

    Returns:
        dict: Dictionary of data in PyTorch tensors.

    """
    for key, value in array_dict.items():
        array_dict[key] = torch.FloatTensor(value)
    return array_dict


def torch_to_np(value_in):
    """Convert PyTorch tensors to numpy arrays.

    Args:
        value_in (tuple): Tuple of data in PyTorch tensors.

    Returns:
        tuple[numpy.ndarray]: Tuple of data in numpy arrays.

    """
    value_out = []
    for v in value_in:
        value_out.append(v.numpy())
    return tuple(value_out)


def flatten_batch(tensor):
    """Flatten a batch of observations.

    Reshape a tensor of size (X, Y, Z) into (X*Y, Z)

    Args:
        tensor (torch.Tensor): Tensor to flatten.

    Returns:
        torch.Tensor: Flattened tensor.

    """
    return tensor.reshape((-1, ) + tensor.shape[2:])
