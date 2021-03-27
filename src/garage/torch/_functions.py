"""Utility functions for PyTorch algorithms.

A collection of common functions that are used by Pytorch algos.

This collection of functions can be used to manage the following:
    - Pytorch GPU usage
        - setting the default Pytorch GPU
        - converting Tensors to GPU Tensors
        - Converting Tensors into `numpy.ndarray` format and vice versa
    - Updating model parameters
"""
import copy
import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F

_USE_GPU = False
_DEVICE = None
_GPU_ID = 0


def zero_optim_grads(optim, set_to_none=True):
    """Sets the gradient of all optimized tensors to None.

    This is an optimization alternative to calling `optimizer.zero_grad()`

    Args:
        optim (torch.nn.Optimizer): The optimizer instance
            to zero parameter gradients.
        set_to_none (bool): Set gradients to None
            instead of calling `zero_grad()`which
            sets to 0.
    """
    if not set_to_none:
        optim.zero_grad()
        return

    for group in optim.param_groups:
        for param in group['params']:
            param.grad = None


def compute_advantages(discount, gae_lambda, max_episode_length, baselines,
                       rewards):
    """Calculate advantages.

    Advantages are a discounted cumulative sum.

    Calculate advantages using a baseline according to Generalized Advantage
    Estimation (GAE)

    The discounted cumulative sum can be computed using conv2d with filter.
    filter:
        [1, (discount * gae_lambda), (discount * gae_lambda) ^ 2, ...]
        where the length is same with max_episode_length.

    baselines and rewards are also has same shape.
        baselines:
        [ [b_11, b_12, b_13, ... b_1n],
          [b_21, b_22, b_23, ... b_2n],
          ...
          [b_m1, b_m2, b_m3, ... b_mn] ]
        rewards:
        [ [r_11, r_12, r_13, ... r_1n],
          [r_21, r_22, r_23, ... r_2n],
          ...
          [r_m1, r_m2, r_m3, ... r_mn] ]

    Args:
        discount (float): RL discount factor (i.e. gamma).
        gae_lambda (float): Lambda, as used for Generalized Advantage
            Estimation (GAE).
        max_episode_length (int): Maximum length of a single episode.
        baselines (torch.Tensor): A 2D vector of value function estimates with
            shape (N, T), where N is the batch dimension (number of episodes)
            and T is the maximum episode length experienced by the agent. If an
            episode terminates in fewer than T time steps, the remaining
            elements in that episode should be set to 0.
        rewards (torch.Tensor): A 2D vector of per-step rewards with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum episode length experienced by the agent. If an
            episode terminates in fewer than T time steps, the remaining
            elements in that episode should be set to 0.

    Returns:
        torch.Tensor: A 2D vector of calculated advantage values with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum episode length experienced by the agent. If an
            episode terminates in fewer than T time steps, the remaining values
            in that episode should be set to 0.

    """
    adv_filter = torch.full((1, 1, 1, max_episode_length - 1),
                            discount * gae_lambda,
                            dtype=torch.float)
    adv_filter = torch.cumprod(F.pad(adv_filter, (1, 0), value=1), dim=-1)

    deltas = (rewards + discount * F.pad(baselines, (0, 1))[:, 1:] - baselines)
    deltas = F.pad(deltas,
                   (0, max_episode_length - 1)).unsqueeze(0).unsqueeze(0)

    advantages = F.conv2d(deltas, adv_filter, stride=1).reshape(rewards.shape)
    return advantages


def pad_to_last(nums, total_length, axis=-1, val=0):
    """Pad val to last in nums in given axis.

    length of the result in given axis should be total_length.

    Raises:
      IndexError: If the input axis value is out of range of the nums array

    Args:
        nums (numpy.ndarray): The array to pad.
        total_length (int): The final width of the Array.
        axis (int): Axis along which a sum is performed.
        val (int): The value to set the padded value.

    Returns:
        torch.Tensor: Padded array

    """
    tensor = torch.Tensor(nums)
    axis = (axis + len(tensor.shape)) if axis < 0 else axis

    if len(tensor.shape) <= axis:
        raise IndexError('axis {} is out of range {}'.format(
            axis, tensor.shape))

    padding_config = [0, 0] * len(tensor.shape)
    padding_idx = abs(axis - len(tensor.shape)) * 2 - 1
    padding_config[padding_idx] = max(total_length - tensor.shape[axis], val)
    return F.pad(tensor, padding_config)


def filter_valids(tensor, valids):
    """Filter out tensor using valids (last index of valid tensors).

    valids contains last indices of each rows.

    Args:
        tensor (torch.Tensor): The tensor to filter
        valids (list[int]): Array of length of the valid values

    Returns:
        torch.Tensor: Filtered Tensor

    """
    return [tensor[i][:valid] for i, valid in enumerate(valids)]


def np_to_torch(array):
    """Numpy arrays to PyTorch tensors.

    Args:
        array (np.ndarray): Data in numpy array.

    Returns:
        torch.Tensor: float tensor on the global device.

    """
    tensor = torch.from_numpy(array)

    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    return tensor.to(global_device())


def list_to_tensor(data):
    """Convert a list to a PyTorch tensor.

    Args:
        data (list): Data to convert to tensor

    Returns:
        torch.Tensor: A float tensor
    """
    return torch.as_tensor(data, dtype=torch.float32, device=global_device())


def as_torch_dict(array_dict):
    """Convert a dict whose values are numpy arrays to PyTorch tensors.

    Modifies array_dict in place.

    Args:
        array_dict (dict): Dictionary of data in numpy arrays

    Returns:
        dict: Dictionary of data in PyTorch tensors

    """
    for key, value in array_dict.items():
        array_dict[key] = np_to_torch(value)
    return array_dict


def torch_to_np(tensors):
    """Convert PyTorch tensors to numpy arrays.

    Args:
        tensors (tuple): Tuple of data in PyTorch tensors.

    Returns:
        tuple[numpy.ndarray]: Tuple of data in numpy arrays.

    Note: This method is deprecated and now replaced by
        `garage.torch._functions.to_numpy`.

    """
    value_out = tuple(v.cpu().numpy() for v in tensors)
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


def flatten_to_single_vector(tensor):
    """Collapse the C x H x W values per representation into a single long vector.

    Reshape a tensor of size (N, C, H, W) into (N, C * H * W).

    Args:
        tensor (torch.tensor): batch of data.

    Returns:
        torch.Tensor: Reshaped view of that data (analogous to numpy.reshape)

    """
    N = tensor.shape[0]  # read in N, C, H, W
    return tensor.view(N, -1)


def update_module_params(module, new_params):  # noqa: D202
    """Load parameters to a module.

    This function acts like `torch.nn.Module._load_from_state_dict()`, but
    it replaces the tensors in module with those in new_params, while
    `_load_from_state_dict()` loads only the value. Use this function so
    that the `grad` and `grad_fn` of `new_params` can be restored

    Args:
        module (torch.nn.Module): A torch module.
        new_params (dict): A dict of torch tensor used as the new
            parameters of this module. This parameters dict should be
            generated by `torch.nn.Module.named_parameters()`

    """
    named_modules = dict(module.named_modules())

    # pylint: disable=protected-access
    def update(m, name, param):
        del m._parameters[name]  # noqa: E501
        setattr(m, name, param)
        m._parameters[name] = param  # noqa: E501

    for name, new_param in new_params.items():
        if '.' in name:
            module_name, param_name = tuple(name.rsplit('.', 1))
            if module_name in named_modules:
                update(named_modules[module_name], param_name, new_param)
        else:
            update(module, name, new_param)


# pylint: disable=missing-param-doc, missing-type-doc
def soft_update_model(target_model, source_model, tau):
    """Update model parameter of target and source model.

    # noqa: D417
    Args:
        target_model
                (garage.torch.Policy/garage.torch.QFunction):
                    Target model to update.
        source_model
                (garage.torch.Policy/QFunction):
                    Source network to update.
        tau (float): Interpolation parameter for doing the
            soft target update.

    """
    for target_param, param in zip(target_model.parameters(),
                                   source_model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def set_gpu_mode(mode, gpu_id=0):
    """Set GPU mode and device ID.

    Args:
        mode (bool): Whether or not to use GPU
        gpu_id (int): GPU ID

    """
    # pylint: disable=global-statement
    global _GPU_ID
    global _USE_GPU
    global _DEVICE
    _GPU_ID = gpu_id
    _USE_GPU = mode
    _DEVICE = torch.device(('cuda:' + str(_GPU_ID)) if _USE_GPU else 'cpu')


def prefer_gpu():
    """Prefer to use GPU(s) if GPU(s) is detected."""
    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)


def global_device():
    """Returns the global device that torch.Tensors should be placed on.

    Note: The global device is set by using the function
        `garage.torch._functions.set_gpu_mode.`
        If this functions is never called
        `garage.torch._functions.device()` returns None.

    Returns:
        `torch.Device`: The global device that newly created torch.Tensors
            should be placed on.

    """
    # pylint: disable=global-statement
    global _DEVICE
    return _DEVICE


def product_of_gaussians(mus, sigmas_squared):
    """Compute mu, sigma of product of gaussians.

    Args:
        mus (torch.Tensor): Means, with shape :math:`(N, M)`. M is the number
            of mean values.
        sigmas_squared (torch.Tensor): Variances, with shape :math:`(N, V)`. V
            is the number of variance values.

    Returns:
        torch.Tensor: Mu of product of gaussians, with shape :math:`(N, 1)`.
        torch.Tensor: Sigma of product of gaussians, with shape :math:`(N, 1)`.

    """
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def state_dict_to(state_dict, device):
    """Move optimizer to a specified device.

    Args:
        state_dict (dict): state dictionary to be moved
        device (str): ID of GPU or CPU.

    Returns:
        dict: state dictionary moved to device
    """
    for param in state_dict.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
        elif isinstance(param, dict):
            state_dict_to(param, device)
    return state_dict


# pylint: disable=W0223
class NonLinearity(nn.Module):
    """Wrapper class for non linear function or module.

    Args:
        non_linear (callable or type): Non-linear function or type to be
            wrapped.

    """

    def __init__(self, non_linear):
        super().__init__()

        if isinstance(non_linear, type):
            self.module = non_linear()
        elif callable(non_linear):
            self.module = copy.deepcopy(non_linear)
        else:
            raise ValueError(
                'Non linear function {} is not supported'.format(non_linear))

    # pylint: disable=arguments-differ
    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values

        Returns:
            torch.Tensor: Output value

        """
        return self.module(input_value)

    # pylint: disable=missing-return-doc, missing-return-type-doc
    def __repr__(self):
        """object representation method."""
        return repr(self.module)


def _value_at_axis(value, axis):
    """Get the value for a particular axis.

    Args:
        value (tuple or list or int): Possible tuple of per-axis values.
        axis (int): Axis to get value for.

    Returns:
        int: the value at the available axis.

    """
    if not isinstance(value, (list, tuple)):
        return value
    if len(value) == 1:
        return value[0]
    else:
        return value[axis]


def output_height_2d(layer, height):
    """Compute the output height of a torch.nn.Conv2d, assuming NCHW format.

    This requires knowing the input height. Because NCHW format makes this very
    easy to mix up, this is a seperate function from conv2d_output_height.

    It also works on torch.nn.MaxPool2d.

    This function implements the formula described in the torch.nn.Conv2d
    documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        layer (torch.nn.Conv2d): The layer to compute output size for.
        height (int): The height of the input image.

    Returns:
        int: The height of the output image.

    """
    assert isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d))
    padding = _value_at_axis(layer.padding, 0)
    dilation = _value_at_axis(layer.dilation, 0)
    kernel_size = _value_at_axis(layer.kernel_size, 0)
    stride = _value_at_axis(layer.stride, 0)
    return math.floor((height + 2 * padding - dilation *
                       (kernel_size - 1) - 1) / stride + 1)


def output_width_2d(layer, width):
    """Compute the output width of a torch.nn.Conv2d, assuming NCHW format.

    This requires knowing the input width. Because NCHW format makes this very
    easy to mix up, this is a seperate function from conv2d_output_height.

    It also works on torch.nn.MaxPool2d.

    This function implements the formula described in the torch.nn.Conv2d
    documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        layer (torch.nn.Conv2d): The layer to compute output size for.
        width (int): The width of the input image.

    Returns:
        int: The width of the output image.

    """
    assert isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d))

    padding = _value_at_axis(layer.padding, 1)
    dilation = _value_at_axis(layer.dilation, 1)
    kernel_size = _value_at_axis(layer.kernel_size, 1)
    stride = _value_at_axis(layer.stride, 1)
    return math.floor((width + 2 * padding - dilation *
                       (kernel_size - 1) - 1) / stride + 1)


def expand_var(name, item, n_expected, reference):
    """Expand a variable to an expected length.

    This is used to handle arguments to primitives that can all be reasonably
    set to the same value, or multiple different values.

    Args:
        name (str): Name of variable being expanded.
        item (any): Element being expanded.
        n_expected (int): Number of elements expected.
        reference (str): Source of n_expected.

    Returns:
        list: List of references to item or item itself.

    Raises:
        ValueError: If the variable is a sequence but length of the variable
            is not 1 or n_expected.

    """
    if n_expected == 1:
        warnings.warn(
            f'Providing a {reference} of length 1 prevents {name} from '
            f'being expanded.')
    if isinstance(item, (list, tuple)):
        if len(item) == n_expected:
            return item
        elif len(item) == 1:
            return list(item) * n_expected
        else:
            raise ValueError(
                f'{name} is length {len(item)} but should be length '
                f'{n_expected} to match {reference}')
    else:
        return [item] * n_expected
