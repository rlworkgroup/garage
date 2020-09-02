"""Utility functions for tf-based Reinforcement learning algorithms."""
import collections

import numpy as np
import tensorflow as tf

from garage.np import discount_cumsum


def paths_to_tensors(paths, max_episode_length, baseline_predictions, discount,
                     gae_lambda):
    """Return processed sample data based on the collected paths.

    Args:
        paths (list[dict]): A list of collected paths.
        max_episode_length (int): Maximum length of a single episode.
        baseline_predictions(numpy.ndarray): : Predicted value of GAE
            (Generalized Advantage Estimation) Baseline.
        discount (float): Environment reward discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.

    Returns:
        dict: Processed sample data, with key
            * observations: (numpy.ndarray)
            * actions: (numpy.ndarray)
            * rewards: (numpy.ndarray)
            * baselines: (numpy.ndarray)
            * returns: (numpy.ndarray)
            * valids: (numpy.ndarray)
            * agent_infos: (dict)
            * env_infos: (dict)
            * paths: (list[dict])

    """
    baselines = []
    returns = []
    total_steps = 0

    for idx, path in enumerate(paths):
        total_steps += len(path['rewards'])
        path_baselines = np.append(baseline_predictions[idx], 0)
        deltas = (path['rewards'] + discount * path_baselines[1:] -
                  path_baselines[:-1])
        path['advantages'] = discount_cumsum(deltas, discount * gae_lambda)
        path['deltas'] = deltas

    for idx, path in enumerate(paths):
        # baselines
        path['baselines'] = baseline_predictions[idx]
        baselines.append(path['baselines'])

        # returns
        path['returns'] = discount_cumsum(path['rewards'], discount)
        returns.append(path['returns'])

    # make all paths the same length
    obs = [path['observations'] for path in paths]
    obs = pad_tensor_n(obs, max_episode_length)

    actions = [path['actions'] for path in paths]
    actions = pad_tensor_n(actions, max_episode_length)

    rewards = [path['rewards'] for path in paths]
    rewards = pad_tensor_n(rewards, max_episode_length)

    returns = [path['returns'] for path in paths]
    returns = pad_tensor_n(returns, max_episode_length)

    baselines = pad_tensor_n(baselines, max_episode_length)

    agent_infos = [path['agent_infos'] for path in paths]
    agent_infos = stack_tensor_dict_list(
        [pad_tensor_dict(p, max_episode_length) for p in agent_infos])

    env_infos = [path['env_infos'] for path in paths]
    env_infos = stack_tensor_dict_list(
        [pad_tensor_dict(p, max_episode_length) for p in env_infos])

    valids = [np.ones_like(path['returns']) for path in paths]
    valids = pad_tensor_n(valids, max_episode_length)

    lengths = np.asarray([v.sum() for v in valids])

    samples_data = dict(
        observations=obs,
        actions=actions,
        rewards=rewards,
        baselines=baselines,
        returns=returns,
        valids=valids,
        lengths=lengths,
        agent_infos=agent_infos,
        env_infos=env_infos,
        paths=paths,
    )

    return samples_data


def compile_function(inputs, outputs):
    """Compiles a tensorflow function using the current session.

    Args:
        inputs (list[tf.Tensor]): Inputs to the function. Can be a list of
            inputs or just one.
        outputs (list[tf.Tensor]): Outputs of the function. Can be a list of
            outputs or just one.

    Returns:
        Callable: Compiled TensorFlow function.
    """

    def _run(*input_vals):
        # pylint: disable=missing-return-doc, missing-return-type-doc
        sess = tf.compat.v1.get_default_session()
        return sess.run(outputs, feed_dict=dict(list(zip(inputs, input_vals))))

    return _run


def get_target_ops(variables, target_variables, tau=None):
    """Get target variables update operations.

    In RL algorithms we often update target network every n
    steps. This function returns the tf.Operation for updating
    target variables (denoted by target_var) from variables
    (denote by var) with fraction tau. In other words, each time
    we want to keep tau of the var and add (1 - tau) of target_var
    to var.

    Args:
        variables (list[tf.Variable]): Soure variables for update.
        target_variables (list[tf.Variable]): Target variables to
            be updated.
        tau (float): Fraction to update. Set it to be None for
            hard-update.

    Returns:
        tf.Operation: Operation for updating the target variables.
    """
    update_ops = []
    init_ops = []
    assert len(variables) == len(target_variables)
    for var, target_var in zip(variables, target_variables):
        init_ops.append(tf.compat.v1.assign(target_var, var))

        if tau is not None:
            update_ops.append(
                tf.compat.v1.assign(target_var,
                                    tau * var + (1.0 - tau) * target_var))

    if tau is not None:
        return init_ops, update_ops
    else:
        return init_ops


def flatten_batch(t, name='flatten_batch'):
    """Flatten a batch of observations.

    Reshape a tensor of size (X, Y, Z) into (X*Y, Z)

    Args:
        t (tf.Tensor): Tensor to flatten.
        name (string): Name of the operation.

    Returns:
        tf.Tensor: Flattened tensor.
    """
    return tf.reshape(t, [-1] + list(t.shape[2:]), name=name)


def flatten_batch_dict(d, name='flatten_batch_dict'):
    """Flatten a batch of observations represented as a dict.

    Args:
        d (dict[tf.Tensor]): A dict of Tensors to flatten.
        name (string): The name of the operation (None by default).

    Returns:
        dict[tf.Tensor]: A dict with flattened tensors.
    """
    with tf.name_scope(name):
        return {k: flatten_batch(v) for k, v in d.items()}


def filter_valids(t, valid, name='filter_valids'):
    """Filter out tensor using valid array.

    Args:
        t (tf.Tensor): The tensor to filter.
        valid (list[float]): Array of length of the valid values (either
            0 or 1).
        name (string): Name of the operation.

    Returns:
        tf.Tensor: Filtered Tensor.
    """
    # Must round before cast to prevent floating-error
    return tf.dynamic_partition(t,
                                tf.cast(tf.round(valid), tf.int32),
                                2,
                                name=name)[1]


def filter_valids_dict(d, valid, name='filter_valids_dict'):
    """Filter valid values on a dict.

    Args:
        d (dict[tf.Tensor]): Dict of tensors to be filtered.
        valid (list[float]): Array of length of the valid values (elements
        can be either 0 or 1).
        name (string): Name of the operation. None by default.

    Returns:
        dict[tf.Tensor]: Dict with filtered tensors.
    """
    with tf.name_scope(name):
        return {k: filter_valids(v, valid) for k, v in d.items()}


def graph_inputs(name, **kwargs):
    """Creates a namedtuple of the given keys and values.

    Args:
        name (string): Name of the tuple.
        kwargs (tf.Tensor): One or more tensor(s) to add to the
            namedtuple's values. The parameter names are used as keys
            in the namedtuple. Ex. obs1=tensor1, obs2=tensor2.

    Returns:
        tuple: namedtuple containing the collection of variables passed.
    """
    Singleton = collections.namedtuple(name, kwargs.keys())

    return Singleton(**kwargs)


def flatten_inputs(deep):
    """Flattens an :class:`collections.abc.Iterable` recursively.

    Args:
        deep (Iterable): An :class:`~collections.abc.Iterable` to flatten.

    Returns:
        list: The flattened result.
    """

    def flatten(deep):
        # pylint: disable=missing-yield-doc,missing-yield-type-doc
        for d in deep:
            if isinstance(d, collections.abc.Iterable) and not isinstance(
                    d, (str, bytes, tf.Tensor, np.ndarray)):
                yield from flatten(d)
            else:
                yield d

    return list(flatten(deep))


def flatten_tensor_variables(ts):
    """Flattens a list of tensors into a single, 1-dimensional tensor.

    Args:
        ts (collections.abc.Iterable): Iterable containing either tf.Tensors or
            arrays.

    Returns:
        tf.Tensor: Flattened Tensor.
    """
    return tf.concat([tf.reshape(x, [-1]) for x in ts],
                     0,
                     name='flatten_tensor_variables')


def new_tensor(name, ndim, dtype):
    """Creates a placeholder :class:`tf.Tensor` with the specified arguments.

    Args:
        name (string): Name of the tf.Tensor.
        ndim (int): Number of dimensions of the tf.Tensor.
        dtype (type): Data type of the tf.Tensor's contents.

    Returns:
        tf.Tensor: Placeholder tensor.
    """
    return tf.compat.v1.placeholder(dtype=dtype,
                                    shape=[None] * ndim,
                                    name=name)


def new_tensor_like(name, arr_like):
    """Creates a new placeholder :class:`tf.Tensor` similar to arr_like.

    The new :class:`tf.Tensor` has the same number of dimensions and
    dtype as arr_like.

    Args:
        name (string): Name of the new tf.Tensor.
        arr_like (tf.Tensor): Tensor to copy attributes from.

    Returns:
        tf.Tensor: New placeholder tensor.
    """
    return new_tensor(name,
                      arr_like.get_shape().ndims, arr_like.dtype.base_dtype)


def concat_tensor_list(tensor_list):
    """Concatenates a list of tensors into one tensor.

    Args:
        tensor_list (list[ndarray]): list of tensors.

    Return:
        ndarray: Concatenated tensor.
    """
    return np.concatenate(tensor_list, axis=0)


def concat_tensor_dict_list(tensor_dict_list):
    """Concatenates a dict of tensors lists.

    Each list of tensors gets concatenated into one tensor.

    Args:
        tensor_dict_list (dict[list[ndarray]]): Dict with lists of tensors.

    Returns:
        dict[ndarray]: A dict with the concatenated tensors.
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


def stack_tensor_dict_list(tensor_dict_list):
    """Stack a list of dictionaries of {tensors or dictionary of tensors}.

    Args:
        tensor_dict_list (dict): a list of dictionaries of {tensors or
            dictionary of tensors}.

    Returns:
        dict: a dictionary of {stacked tensors or dictionary of stacked
            tensors}.
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]

        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = np.array([x[k] for x in tensor_dict_list])

        ret[k] = v

    return ret


def split_tensor_dict_list(tensor_dict):
    """Split a list of dictionaries of {tensors or dictionary of tensors}.

    Args:
        tensor_dict (dict): a list of dictionaries of {tensors or
        dictionary of tensors}.

    Returns:
        dict: a dictionary of {split tensors or dictionary of split tensors}.
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


def pad_tensor(x, max_len):
    """Pad tensors with zeros.

    Args:
        x (numpy.ndarray): Tensors to be padded.
        max_len (int): Maximum length.

    Returns:
        numpy.ndarray: Padded tensor.
    """
    return np.concatenate([
        x,
        np.tile(np.zeros_like(x[0]),
                (max_len - len(x), ) + (1, ) * np.ndim(x[0]))
    ])


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


def pad_tensor_dict(tensor_dict, max_len):
    """Pad dictionary of tensors with zeros.

    Args:
        tensor_dict (dict[numpy.ndarray]): Tensors to be padded.
        max_len (int): Maximum length.

    Returns:
        dict[numpy.ndarray]: Padded tensor.
    """
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = pad_tensor_dict(tensor_dict[k], max_len)
        else:
            ret[k] = pad_tensor(tensor_dict[k], max_len)

    return ret


def compute_advantages(discount,
                       gae_lambda,
                       max_len,
                       baselines,
                       rewards,
                       name='compute_advantages'):
    """Calculate advantages.

    Advantages are a discounted cumulative sum.

    The discount cumulative sum can be represented as an IIR
    filter ob the reversed input vectors, i.e.

    y[t] - discount*y[t+1] = x[t], or
    rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]

    Given the time-domain IIR filter step response, we can
    calculate the filter response to our signal by convolving the
    signal with the filter response function. The time-domain IIR
    step response is calculated below as discount_filter:

    discount_filter = [1, discount, discount^2, ..., discount^N-1]
    where the epsiode length is N.

    We convolve discount_filter with the reversed time-domain
    signal deltas to calculate the reversed advantages:

    rev(advantages) = discount_filter (X) rev(deltas)

    TensorFlow's tf.nn.conv1d op is not a true convolution, but
    actually a cross-correlation, so its input and output are
    already implicitly reversed for us.

    advantages = discount_filter (tf.nn.conv1d) deltas

    Args:
        discount (float): Discount factor.
        gae_lambda (float): Lambda, as used for Generalized Advantage
            Estimation (GAE).
        max_len (int): Maximum length of a single episode.
        baselines (tf.Tensor): A 2D vector of value function estimates with
            shape :math:`(N, T)`, where :math:`N` is the batch dimension
            (number of episodes) and :math:`T` is the maximum episode length
            experienced by the agent.
        rewards (tf.Tensor): A 2D vector of per-step rewards with shape
            :math:`(N, T)`, where :math:`N` is the batch dimension (number of
            episodes) and :math:`T` is the maximum episode length experienced
            by the agent.
        name (string): Name of the operation.

    Returns:
        tf.Tensor: A 2D vector of calculated advantage values with shape
            :math:`(N, T)`, where :math:`N` is the batch dimension (number of
            episodes) and :math:`T` is the maximum episode length experienced
            by the agent.
    """
    with tf.name_scope(name):
        # Prepare convolutional IIR filter to calculate advantages
        gamma_lambda = tf.constant(float(discount) * float(gae_lambda),
                                   dtype=tf.float32,
                                   shape=[max_len, 1, 1])
        advantage_filter = tf.compat.v1.cumprod(gamma_lambda,
                                                axis=0,
                                                exclusive=True)

        # Calculate deltas
        pad = tf.zeros_like(baselines[:, :1])
        baseline_shift = tf.concat([baselines[:, 1:], pad], 1)
        deltas = rewards + discount * baseline_shift - baselines

        # Convolve deltas with the discount filter to get advantages
        deltas_pad = tf.expand_dims(tf.concat(
            [deltas, tf.zeros_like(deltas[:, :-1])], 1),
                                    axis=2)
        adv = tf.nn.conv1d(deltas_pad,
                           advantage_filter,
                           stride=1,
                           padding='VALID')
        advantages = tf.reshape(adv, [-1])

    return advantages


def center_advs(advs, axes, eps, offset=0, scale=1, name='center_adv'):
    """Normalize the advs tensor.

    This calculates the mean and variance using the axes specified
    and normalizes the tensor using those values.

    Args:
        advs (tf.Tensor): Tensor to normalize.
        axes (array[int]): Axes along which to compute the mean and variance.
        eps (float): Small number to avoid dividing by zero.
        offset (tf.Tensor): Offset added to the normalized tensor.
            This is zero by default.
        scale (tf.Tensor): Scale to apply to the normalized tensor. This is
            1 by default but can also be None.
        name (string): Name of the operation. None by default.

    Returns:
        tf.Tensor: Normalized, scaled and offset tensor.
    """
    with tf.name_scope(name):
        mean, var = tf.nn.moments(advs, axes=axes)
        advs = tf.nn.batch_normalization(advs, mean, var, offset, scale, eps)

    return advs


def positive_advs(advs, eps, name='positive_adv'):
    """Make all the values in the advs tensor positive.

    Offsets all values in advs by the minimum value in
    the tensor, plus an epsilon value to avoid dividing by zero.

    Args:
        advs (tf.Tensor): The tensor to offset.
        eps (tf.float32): A small value to avoid by-zero division.
        name (string): Name of the operation.

    Returns:
        tf.Tensor: Tensor with modified (postiive) values.
    """
    with tf.name_scope(name):
        m = tf.reduce_min(advs)
        advs = (advs - m) + eps

    return advs


def discounted_returns(discount, max_len, rewards, name='discounted_returns'):
    """Calculate discounted returns.

    Args:
        discount (float): Discount factor.
        max_len (int): Maximum length of a single episode.
        rewards (tf.Tensor): A 2D vector of per-step rewards with shape
            :math:`(N, T)`, where :math:`N` is the batch dimension (number of
            episodes) and :math:`T` is the maximum episode length experienced
            by the agent.
        name (string): Name of the operation. None by default.

    Returns:
        tf.Tensor: Tensor of discounted returns.
    """
    with tf.name_scope(name):
        gamma = tf.constant(float(discount),
                            dtype=tf.float32,
                            shape=[max_len, 1, 1])
        return_filter = tf.math.cumprod(gamma, axis=0, exclusive=True)
        rewards_pad = tf.expand_dims(tf.concat(
            [rewards, tf.zeros_like(rewards[:, :-1])], 1),
                                     axis=2)
        returns = tf.nn.conv1d(rewards_pad,
                               return_filter,
                               stride=1,
                               padding='VALID')

    return returns
