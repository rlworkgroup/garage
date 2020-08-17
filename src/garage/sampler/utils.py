"""Utility functions related to sampling."""

import numpy as np

from garage.misc import tensor_utils


def rollout(env,
            agent,
            *,
            max_episode_length=np.inf,
            animated=False,
            speedup=1,
            deterministic=False):
    """Sample a single episode of the agent in the environment.

    Args:
        agent (Policy): Agent used to select actions.
        env (Environment): Environment to perform actions in.
        max_episode_length (int): If the episode reaches this many timesteps,
            it is truncated.
        animated (bool): If true, render the environment after each step.
        speedup (float): Factor by which to decrease the wait time between
            rendered steps. Only relevant, if animated == true.
        deterministic (bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.

    Returns:
        dict[str, np.ndarray or dict]: Dictionary, with keys:
            * observations(np.array): Flattened array of observations.
                There should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape (T + 1, S^*) (the
                unflattened state space of the current environment).
            * actions(np.array): Non-flattened array of actions. Should have
                shape (T, S^*) (the unflattened action space of the current
                environment).
            * rewards(np.array): Array of rewards of shape (T,) (1D array of
                length timesteps).
            * agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            * env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.
            * dones(np.array): Array of termination signals.

    """
    del speedup
    env_steps = []
    agent_infos = []
    observations = []
    last_obs = env.reset()[0]
    agent.reset()
    episode_length = 0
    if animated:
        env.visualize()
    while episode_length < (max_episode_length or np.inf):
        a, agent_info = agent.get_action(last_obs)
        if deterministic and 'mean' in agent_info:
            a = agent_info['mean']
        es = env.step(a)
        env_steps.append(es)
        observations.append(last_obs)
        agent_infos.append(agent_info)
        episode_length += 1
        if es.last:
            break
        last_obs = es.observation

    return dict(
        observations=np.array(observations),
        actions=np.array([es.action for es in env_steps]),
        rewards=np.array([es.reward for es in env_steps]),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(
            [es.env_info for es in env_steps]),
        dones=np.array([es.terminal for es in env_steps]),
    )


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
                truncated_last_path[k] = tensor_utils.truncate_tensor_dict(
                    v, truncated_len)
            else:
                raise ValueError(
                    'Unexpected key {} found in path. Valid keys: {}'.format(
                        k, valid_keys))
        paths.append(truncated_last_path)
    return paths
