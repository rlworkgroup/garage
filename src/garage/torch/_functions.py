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

import gym
import torch
from torch import nn
import torch.nn.functional as F

_USE_GPU = False
_DEVICE = None
_GPU_ID = 0


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
    return torch.from_numpy(array).float().to(global_device())


def dict_np_to_torch(array_dict):
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
    value_out = tuple(v.numpy() for v in tensors)
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

    # pylint: disable=protected-access
    def update(m, name, param):
        del m._parameters[name]  # noqa: E501
        setattr(m, name, param)
        m._parameters[name] = param  # noqa: E501

    named_modules = dict(module.named_modules())

    for name, new_param in new_params.items():
        if '.' in name:
            module_name, param_name = tuple(name.rsplit('.', 1))
            if module_name in named_modules:
                update(named_modules[module_name], param_name, new_param)
        else:
            update(module, name, new_param)


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


class TransposeImage(gym.ObservationWrapper):
    """Transpose observation space for image observation in PyTorch."""

    def __init__(self, env=None):
        """Transpose image observation space.

        Reshape the input observation shape from (H, W, C) into (C, H, W)
        in pytorch format.

        Args:
            env (Environment): environment.
        """
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        """Transpose image observation.

        Args:
            observation (tensor): observation.

        Returns:
            torch.Tensor: transposed observation.
        """
        # Observation is of type Tensor
        return observation.transpose(2, 0, 1)
