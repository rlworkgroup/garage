"""Utiliy functions for tensors."""
import gym.spaces
import numpy as np
import scipy.signal


def discount_cumsum(x, discount):
    """Discounted cumulative sum.

    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501
    Here, we have y[t] - discount*y[t+1] = x[t]
    or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]

    Args:
        x (np.ndarrary): Input.
        discount (float): Discount factor.

    Returns:
        np.ndarrary: Discounted cumulative sum.

    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1],
                                axis=0)[::-1]


def explained_variance_1d(ypred, y, valids):
    """Explained variation for 1D inputs.

    It is the proportion of the variance in one variable that is explained or
    predicted from another variable.

    Args:
        ypred (np.ndarray): Sample data from the first variable.
            Shape: :math:`(N, max_path_length)`.
        y (np.ndarray): Sample data from the second variable.
            Shape: :math:`(N, max_path_length)`.
        valids (np.ndarray): Array indicating valid indices.
            Shape: :math:`(N, max_path_length)`.

    Returns:
        float: The explained variance.

    """
    ypred = ypred[valids.astype(np.bool)]
    y = y[valids.astype(np.bool)]
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 0:
            return 0
        return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)


def flatten_tensors(tensors):
    """Flatten a list of tensors.

    Args:
        tensors (list[numpy.ndarray]): List of tensors to be flattened.

    Returns:
        numpy.ndarray: Flattened tensors.

    """
    if tensors:
        return np.concatenate([np.reshape(x, [-1]) for x in tensors])
    return np.asarray([])


def unflatten_tensors(flattened, tensor_shapes):
    """Unflatten a flattened tensors into a list of tensors.

    Args:
        flattened (numpy.ndarray): Flattened tensors.
        tensor_shapes (tuple): Tensor shapes.

    Returns:
        list[numpy.ndarray]: Unflattened list of tensors.

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
        dict_list = [x[k] if k in x else [] for x in tensor_dict_list]
        if isinstance(example, dict):
            v = stack_tensor_dict_list(dict_list)
        else:
            v = np.array(dict_list)
        ret[k] = v
    return ret


def stack_and_pad_tensor_n(paths, key, max_len):
    """Stack and pad array of list of tensors.

    Input paths are a list of N dicts, each with values of shape
    :math:`(D, S^*)`. This function stack and pad the values with the input
    key with max_len, so output will be shape :math:`(N, D, S^*)`.

    Args:
        paths (list[dict]): List of dict to be stacked and padded.
            Value of each dict will be shape of :math:`(D, S^*)`.
        key (str): Key of the values in the paths to be stacked and padded.
        max_len (int): Maximum length for padding.

    Returns:
        numpy.ndarray: Stacked and padded tensor. Shape: :math:`(N, D, S^*)`
            where N is the len of input paths.

    """
    ret = [path[key] for path in paths]
    if isinstance(ret[0], dict):
        ret = stack_tensor_dict_list(
            [pad_tensor_dict(p, max_len) for p in ret])
    else:
        ret = pad_tensor_n(ret, max_len)
    return ret


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
        dict_list = [x[k] if k in x else [] for x in tensor_dict_list]
        if isinstance(example, dict):
            v = concat_tensor_dict_list(dict_list)
        else:
            v = np.concatenate(dict_list, axis=0)
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
            ret[k] = v[:truncated_len]
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


def slice_nested_dict(dict_or_array, start, stop):
    """Slice a dictionary containing arrays (or dictionaries).

    This function is primarily intended for un-batching env_infos and
    action_infos.

    Args:
        dict_or_array (dict[str, dict or np.ndarray] or np.ndarray): A nested
            dictionary should only contain dictionaries and numpy arrays
            (recursively).
        start (int): First index to be included in the slice.
        stop (int): First index to be excluded from the slice. In other words,
            these are typical python slice indices.

    Returns:
        dict or np.ndarray: The input, but sliced.

    """
    if isinstance(dict_or_array, dict):
        return {
            k: slice_nested_dict(v, start, stop)
            for (k, v) in dict_or_array.items()
        }
    else:
        # It *should* be a numpy array (unless someone ignored the type
        # signature).
        return dict_or_array[start:stop]
