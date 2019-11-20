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
    value_out = tuple(v.numpy() for v in value_in)
    return value_out


def flatten_batch(tensor):
    """Flatten a batch of observations.

    Reshape a tensor of size (X, Y, Z) into (X*Y, Z)

    Args:
        tensor (torch.Tensor): Tensor to flatten.

    Returns:
        torch.Tensor: Flattened tensor.

    """
    return tensor.reshape((-1, ) + tensor.shape[2:])


def zero_grad(params):
    for param in params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def update_module_params(module, new_params):
    def update(m, name, param):
        del m._parameters[name]  # pylint: disable=protected-access # noqa: E501
        setattr(m, name, param)
        m._parameters[name] = param  # pylint: disable=protected-access # noqa: E501

    named_modules = dict(module.named_modules())

    for name, new_param in new_params.items():
        if '.' in name:
            module_name, param_name = tuple(name.rsplit('.', 1))
            if module_name in named_modules:
                update(named_modules[module_name], param_name, new_param)
        else:
            update(module, name, new_param)
