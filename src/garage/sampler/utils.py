"""Utility functions related to sampling."""

import time

import numpy as np

from garage.misc import tensor_utils


def rollout(env,
            agent,
            *,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            deterministic=False):
    """Sample a single rollout of the agent in the environment.

    Args:
        agent(Policy): Agent used to select actions.
        env(gym.Env): Environment to perform actions in.
        max_path_length(int): If the rollout reaches this many timesteps, it is
            terminated.
        animated(bool): If true, render the environment after each step.
        speedup(float): Factor by which to decrease the wait time between
            rendered steps. Only relevant, if animated == true.
        deterministic (bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.

    Returns:
        worker_number(int): The worker number passed into this function.
        observations(np.array): Non-flattened array of observations.
        actions(np.array): Non-flattened array of actions.
        rewards(np.array): Array of rewards of shape (timesteps, 1).
        agent_infos(dict[str, np.array]): Dictionary of stacked, non-flattened
            `agent_info`s.
        env_infos(dict[str, np.array]): Dictionary of stacked, non-flattened
            `env_info`s.

    """
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        if deterministic:
            a = agent_info['mean']
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )


def truncate_paths(paths, max_samples):
    """Truncate the paths so that the total number of samples is max_samples.

    This is done by removing extra paths at the end of
    the list, and make the last path shorter if necessary

    Args:
      paths([dict[str, np.array]])
      max_samples(int)

    Returns:
      A list of paths, truncated so that the number of samples adds up to
      max-samples>

    """
    # chop samples collected by extra paths
    # make a copy
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
                truncated_last_path[k] = tensor_utils.truncate_tensor_list(
                    v, truncated_len)
            elif k in ['env_infos', 'agent_infos']:
                truncated_last_path[k] = tensor_utils.truncate_tensor_dict(
                    v, truncated_len)
            else:
                raise NotImplementedError
        paths.append(truncated_last_path)
    return paths
