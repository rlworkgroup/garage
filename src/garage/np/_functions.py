"""Utility functions for NumPy-based Reinforcement learning algorithms."""
import numpy as np

from garage._dtypes import TrajectoryBatch
from garage.misc import tensor_utils
from garage.sampler.utils import rollout


def samples_to_tensors(paths):
    """Return processed sample data based on the collected paths.

    Args:
        paths (list[dict]): A list of collected paths.

    Returns:
        dict: Processed sample data, with keys
            * undiscounted_returns (list[float])
            * success_history (list[float])
            * complete (list[bool])

    """
    success_history = [
        path['success_count'] / path['running_length'] for path in paths
    ]
    undiscounted_returns = [path['undiscounted_return'] for path in paths]

    # check if the last path is complete
    complete = [path['dones'][-1] for path in paths]

    samples_data = dict(undiscounted_returns=undiscounted_returns,
                        success_history=success_history,
                        complete=complete)

    return samples_data


def obtain_evaluation_samples(policy, env, max_path_length=1000,
                              num_trajs=100):
    """Sample the policy for num_trajs trajectories and return average values.

    Args:
        policy (garage.Policy): Policy to use as the actor when
            gathering samples.
        env (garage.envs.GarageEnv): The environement used to obtain
            trajectories.
        max_path_length (int): Maximum path length. The episode will
            terminate when length of trajectory reaches max_path_length.
        num_trajs (int): Number of trajectories.

    Returns:
        TrajectoryBatch: Evaluation trajectories, representing the best
            current performance of the algorithm.

    """
    paths = []
    # Use a finite length rollout for evaluation.

    for _ in range(num_trajs):
        path = rollout(env,
                       policy,
                       max_path_length=max_path_length,
                       deterministic=True)
        paths.append(path)
    return TrajectoryBatch.from_trajectory_list(env.spec, paths)


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
