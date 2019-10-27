"""Utility functions for tensors."""
import gym.spaces
import numpy as np


def flatten_tensors(tensors):
    """Flatten tensors.

    Args:
        tensors (numpy.ndarray): Tensors to be flattened.

    Returns:
        numpy.ndarray: Flattened tensors.

    """
    if tensors:
        return np.concatenate([np.reshape(x, [-1]) for x in tensors])
    return np.asarray([])


def unflatten_tensors(flattened, tensor_shapes):
    """Unflatten tensors.

    Args:
        flattened (numpy.ndarray): Flattened tensors.
        tensor_shapes (tuple): Tensor shapes.

    Returns:
        numpy.ndarray: Unflattened tensors.

    """
    tensor_sizes = list(map(np.prod, tensor_shapes))
    indices = np.cumsum(tensor_sizes)[:-1]
    return [
        np.reshape(pair[0], pair[1])
        for pair in zip(np.split(flattened, indices), tensor_shapes)
    ]


def pad_tensor(x, max_len, mode='zero'):
    """Pad tensors.

    Args:
        x (numpy.ndarray): Tensors to be padded.
        max_len (int): Maximum length.
        mode (str): If 'last', pad with the last element, otherwise pad with 0.

    Returns:
        numpy.ndarray: Padded tensor.

    """
    padding = np.zeros_like(x[0])
    if mode == 'last':
        padding = x[-1]
    return np.concatenate(
        [x, np.tile(padding, (max_len - len(x), ) + (1, ) * np.ndim(x[0]))])


def pad_tensor_n(xs, max_len):
    """Pad array of tensors.

    Args:
        xs (numpy.ndarray): Tensors to be padded.
        max_len (int): Maximum length.

    Returns:
        numpy.ndarray: Padded tensor.

    """
    ret = np.zeros((len(xs), max_len) + xs[0].shape[1:], dtype=xs[0].dtype)
    for idx, x in enumerate(xs):
        ret[idx][:len(x)] = x
    return ret


def pad_tensor_dict(tensor_dict, max_len, mode='zero'):
    """Pad dictionary of tensors.

    Args:
        tensor_dict (dict[numpy.ndarray]): Tensors to be padded.
        max_len (int): Maximum length.
        mode (str): If 'last', pad with the last element, otherwise pad with 0.

    Returns:
        dict[numpy.ndarray]: Padded tensor.

    """
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = pad_tensor_dict(tensor_dict[k], max_len, mode=mode)
        else:
            ret[k] = pad_tensor(tensor_dict[k], max_len, mode=mode)
    return ret


def stack_tensor_list(tensor_list):
    """Stack tensor list.

    Args:
        tensor_list (list[numpy.ndarray]): List of tensors.

    Returns:
        numpy.ndarray: Stacked list of tensor.

    """
    return np.array(tensor_list)


def stack_tensor_dict_list(tensor_dict_list):
    """Stack a list of dictionaries of {tensors or dictionary of tensors}.

    Args:
        tensor_dict_list (dict[list]): a list of dictionaries of {tensors or
            dictionary of tensors}.

    Return:
        dict: a dictionary of {stacked tensors or dictionary of
            stacked tensors}

    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def concat_tensor_list(tensor_list):
    """Concatenate list of tensor.

    Args:
        tensor_list (list[numpy.ndarray]): List of tensors.

    Returns:
        numpy.ndarray: list of tensor.

    """
    return np.concatenate(tensor_list, axis=0)


def concat_tensor_dict_list(tensor_dict_list):
    """Concatenate dictionary of list of tensor.

    Args:
        tensor_dict_list (dict[list]): a list of dictionaries of {tensors or
            dictionary of tensors}.

    Return:
        dict: a dictionary of {stacked tensors or dictionary of
            stacked tensors}

    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = concat_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def split_tensor_dict_list(tensor_dict):
    """Split dictionary of list of tensor.

    Args:
        tensor_dict (dict[numpy.ndarray]): a dictionary of {tensors or
            dictionary of tensors}.

    Return:
        dict: a dictionary of {stacked tensors or dictionary of
            stacked tensors}

    """
    keys = list(tensor_dict.keys())
    ret = None
    for k in keys:
        vals = tensor_dict[k]
        if isinstance(vals, dict):
            vals = split_tensor_dict_list(vals)
        if ret is None:
            ret = [{k: v} for v in vals]
        else:
            for v, cur_dict in zip(vals, ret):
                cur_dict[k] = v
    return ret


def truncate_tensor_list(tensor_list, truncated_len):
    """Truncate list of tensor.

    Args:
        tensor_list (list[numpy.ndarray]): List of tensors.
        truncated_len (int): Length to truncate.

    Returns:
        numpy.ndarray: list of tensor.

    """
    return tensor_list[:truncated_len]


def truncate_tensor_dict(tensor_dict, truncated_len):
    """Truncate dictionary of list of tensor.

    Args:
        tensor_dict (dict[numpy.ndarray]): a dictionary of {tensors or
            dictionary of tensors}.
        truncated_len (int): Length to truncate.

    Return:
        dict: a dictionary of {stacked tensors or dictionary of
            stacked tensors}

    """
    ret = dict()
    for k, v in tensor_dict.items():
        if isinstance(v, dict):
            ret[k] = truncate_tensor_dict(v, truncated_len)
        else:
            ret[k] = truncate_tensor_list(v, truncated_len)
    return ret


def normalize_pixel_batch(env_spec, observations):
    """Normalize the observations (images).

    If the input are images, it normalized into range [0, 1].

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        observations (numpy.ndarray): Observations from environment.

    Returns:
        numpy.ndarray: Normalized observations.

    """
    if isinstance(env_spec.observation_space, gym.spaces.Box):
        if len(env_spec.observation_space.shape) == 3:
            return [obs.astype(np.float32) / 255.0 for obs in observations]
    return observations


def flatten_batch(obs):
    """Flatten input along batch dimension.

    Args:
        obs (numpy.ndarray): Input to be flattened.

    Returns:
        numpy.ndarray: Flattened input.

    """
    return obs.reshape([-1] + list(obs.shape[2:]))


def filter_valids(obs, valids):
    """Filter input with valids.

    Args:
        obs (numpy.ndarray): Input to be filtered.
        valids (numpy.ndarray): Array with element either 0 or 1. The value of
            i-th element being 1 indicates i-th index is valid.

    Returns:
        numpy.ndarray: Valided input.

    """
    return obs[valids.astype(np.int32).flatten() == 1]
