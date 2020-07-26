"""Utility functions related to sampling."""

from garage.np import truncate_tensor_dict


def truncate_paths(paths, max_samples):
    """Truncate the paths so that the total number of samples is max_samples.

    This is done by removing extra paths at the end of
    the list, and make the last path shorter if necessary

    Args:
        paths (list[dict[str, np.ndarray]]): Samples, items with keys:
            * observations (np.ndarray): Enviroment observations
            * actions (np.ndarray): Agent actions
            * rewards (np.ndarray): Environment rewards
            * env_infos (dict): Environment state information
            * agent_infos (dict): Agent state information
        max_samples(int) : Maximum number of samples allowed.

    Returns:
        list[dict[str, np.ndarray]]: A list of paths, truncated so that the
            number of samples adds up to max-samples

    Raises:
        ValueError: If key a other than 'observations', 'actions', 'rewards',
            'env_infos' and 'agent_infos' is found.

    """
    # chop samples collected by extra paths
    # make a copy
    valid_keys = {
        'observations', 'actions', 'rewards', 'env_infos', 'agent_infos'
    }
    paths = list(paths)
    total_n_samples = sum(len(path['rewards']) for path in paths)

    while paths and total_n_samples - len(paths[-1]['rewards']) >= max_samples:
        total_n_samples -= len(paths.pop(-1)['rewards'])

    if paths:
        last_path = paths.pop(-1)
        truncated_last_path = dict()
        truncated_len = len(
            last_path['rewards']) - (total_n_samples - max_samples)

        for k, v in last_path.items():
            if k in ['observations', 'actions', 'rewards']:
                truncated_last_path[k] = v[:truncated_len]
            elif k in ['env_infos', 'agent_infos']:
                truncated_last_path[k] = truncate_tensor_dict(v, truncated_len)
            else:
                raise ValueError(
                    'Unexpected key {} found in path. Valid keys: {}'.format(
                        k, valid_keys))

        paths.append(truncated_last_path)

    return paths
