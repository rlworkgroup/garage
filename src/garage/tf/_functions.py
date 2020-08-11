"""Utility functions for tf-based Reinforcement learning algorithms."""
import numpy as np

from garage.misc import tensor_utils as np_tensor_utils
from garage.tf.misc import tensor_utils


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
        path['advantages'] = np_tensor_utils.discount_cumsum(
            deltas, discount * gae_lambda)
        path['deltas'] = deltas

    for idx, path in enumerate(paths):
        # baselines
        path['baselines'] = baseline_predictions[idx]
        baselines.append(path['baselines'])

        # returns
        path['returns'] = np_tensor_utils.discount_cumsum(
            path['rewards'], discount)
        returns.append(path['returns'])

    # make all paths the same length
    obs = [path['observations'] for path in paths]
    obs = tensor_utils.pad_tensor_n(obs, max_episode_length)

    actions = [path['actions'] for path in paths]
    actions = tensor_utils.pad_tensor_n(actions, max_episode_length)

    rewards = [path['rewards'] for path in paths]
    rewards = tensor_utils.pad_tensor_n(rewards, max_episode_length)

    returns = [path['returns'] for path in paths]
    returns = tensor_utils.pad_tensor_n(returns, max_episode_length)

    baselines = tensor_utils.pad_tensor_n(baselines, max_episode_length)

    agent_infos = [path['agent_infos'] for path in paths]
    agent_infos = tensor_utils.stack_tensor_dict_list([
        tensor_utils.pad_tensor_dict(p, max_episode_length)
        for p in agent_infos
    ])

    env_infos = [path['env_infos'] for path in paths]
    env_infos = tensor_utils.stack_tensor_dict_list([
        tensor_utils.pad_tensor_dict(p, max_episode_length) for p in env_infos
    ])

    valids = [np.ones_like(path['returns']) for path in paths]
    valids = tensor_utils.pad_tensor_n(valids, max_episode_length)

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
