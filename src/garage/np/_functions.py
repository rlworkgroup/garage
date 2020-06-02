"""Utility functions for NumPy-based Reinforcement learning algorithms."""
import numpy as np

from garage.misc import tensor_utils


def paths_to_tensors(paths, max_path_length, baseline_predictions, discount):
    """Return processed sample data based on the collected paths.

    Args:
        paths (list[dict]): A list of collected paths.
        max_path_length (int): Maximum length of a single rollout.
        baseline_predictions(numpy.ndarray): : Predicted value of GAE
            (Generalized Advantage Estimation) Baseline.
        discount (float): Environment reward discount.

    Returns:
        dict: Processed sample data, with key
            * observations (numpy.ndarray): Padded array of the observations of
                the environment
            * actions (numpy.ndarray): Padded array of the actions fed to the
                the environment
            * rewards (numpy.ndarray): Padded array of the acquired rewards
            * agent_infos (dict): a dictionary of {stacked tensors or
                dictionary of stacked tensors}
            * env_infos (dict): a dictionary of {stacked tensors or
                dictionary of stacked tensors}
            * rewards (numpy.ndarray): Padded array of the validity information


    """
    baselines = []
    returns = []

    for idx, path in enumerate(paths):
        # baselines
        path['baselines'] = baseline_predictions[idx]
        baselines.append(path['baselines'])

        # returns
        path['returns'] = tensor_utils.discount_cumsum(path['rewards'],
                                                       discount)
        returns.append(path['returns'])

    obs = [path['observations'] for path in paths]
    obs = tensor_utils.pad_tensor_n(obs, max_path_length)

    actions = [path['actions'] for path in paths]
    actions = tensor_utils.pad_tensor_n(actions, max_path_length)

    rewards = [path['rewards'] for path in paths]
    rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

    agent_infos = [path['agent_infos'] for path in paths]
    agent_infos = tensor_utils.stack_tensor_dict_list([
        tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos
    ])

    env_infos = [path['env_infos'] for path in paths]
    env_infos = tensor_utils.stack_tensor_dict_list(
        [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos])

    valids = [np.ones_like(path['returns']) for path in paths]
    valids = tensor_utils.pad_tensor_n(valids, max_path_length)

    samples_data = dict(observations=obs,
                        actions=actions,
                        rewards=rewards,
                        agent_infos=agent_infos,
                        env_infos=env_infos,
                        valids=valids)

    return samples_data
