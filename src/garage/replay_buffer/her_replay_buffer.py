"""This module implements a Hindsight Experience Replay (HER).

See: https://arxiv.org/abs/1707.01495.
"""
import copy

import numpy as np

from garage.replay_buffer.path_buffer import PathBuffer


class HERReplayBuffer(PathBuffer):
    """Replay buffer for HER (Hindsight Experience Replay).

    It constructs hindsight examples using future strategy.

    Args:
        replay_k (int): Number of HER transitions to add for each regular
            Transition. Setting this to 0 means that no HER replays will
            be added.
        reward_fn (callable): Function to re-compute the reward with
            substituted goals.
        capacity_in_transitions (int): total size of transitions in the buffer.
        env_spec (EnvSpec): Environment specification.
    """

    def __init__(self, replay_k, reward_fn, capacity_in_transitions, env_spec):
        self._replay_k = replay_k
        self._reward_fn = reward_fn
        self._env_spec = env_spec

        if not float(replay_k).is_integer() or replay_k < 0:
            raise ValueError('replay_k must be an integer and >= 0.')
        super().__init__(capacity_in_transitions)

    def _sample_her_goals(self, path, transition_idx):
        """Samples HER goals from the given path.

        Goals are randomly sampled starting from the index after
        transition_idx in the given path.

        Args:
            path (dict[str, np.ndarray]): A dict containing the transition
                keys, where each key contains an ndarray of shape
                :math:`(T, S^*)`.
            transition_idx (int): index of the current transition. Only
                transitions after the current transitions will be randomly
                sampled for HER goals.

        Returns:
            np.ndarray: A numpy array of HER goals with shape
                (replay_k, goal_dim).

        """
        goal_indexes = np.random.randint(transition_idx + 1,
                                         len(path['observations']),
                                         size=self._replay_k)
        return [
            goal['achieved_goal']
            for goal in np.asarray(path['observations'])[goal_indexes]
        ]

    def _flatten_dicts(self, path):
        for key in ['observations', 'next_observations']:
            if not isinstance(path[key], dict):
                path[key] = self._env_spec.observation_space.flatten_n(
                    path[key])
            else:
                path[key] = self._env_spec.observation_space.flatten(path[key])

    def add_path(self, path):
        """Adds a path to the replay buffer.

        For each transition in the given path except the last one,
        replay_k HER transitions will added to the buffer in addition
        to the one in the path. The last transition is added without
        sampling additional HER goals.

        Args:
            path(dict[str, np.ndarray]): Each key in the dict must map
                to a np.ndarray of shape :math:`(T, S^*)`.

        """
        obs_space = self._env_spec.observation_space
        if not isinstance(path['observations'][0], dict):
            # unflatten dicts if they've been flattened
            path['observations'] = obs_space.unflatten_n(path['observations'])
            path['next_observations'] = (obs_space.unflatten_n(
                path['next_observations']))

        # create HER transitions and add them to the buffer
        for idx in range(path['actions'].shape[0] - 1):
            transition = {key: sample[idx] for key, sample in path.items()}
            her_goals = self._sample_her_goals(path, idx)

            # create replay_k transitions using the HER goals
            for goal in her_goals:

                t_new = copy.deepcopy(transition)
                a_g = t_new['next_observations']['achieved_goal']

                t_new['rewards'] = np.array(self._reward_fn(a_g, goal, None))
                t_new['observations']['desired_goal'] = goal
                t_new['next_observations']['desired_goal'] = copy.deepcopy(
                    goal)
                t_new['terminals'] = np.array(False)

                # flatten the observation dicts now that we're done with them
                self._flatten_dicts(t_new)

                for key in t_new.keys():
                    t_new[key] = t_new[key].reshape(1, -1)

                # Since we're using a PathBuffer, add each new transition
                # as its own path.
                super().add_path(t_new)

        self._flatten_dicts(path)
        super().add_path(path)

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        self.__dict__ = state
