"""This module implements a Hindsight Experience Replay (HER).

See: https://arxiv.org/abs/1707.01495.
"""
import copy

import numpy as np

from garage.replay_buffer.path_buffer import PathBuffer


class HerReplayBuffer(PathBuffer):
    """Replay buffer for HER (Hindsight Experience Replay).

    It constructs hindsight examples using future strategy.

    Args:
        replay_k (int): Number of HER transitions to add for each regular
            Transition. Setting this to 0 means that no HER replays will
            be added.
        reward_fun (callable): Function to re-compute the reward with
            substituted goals.
        capacity_in_transitions (int): total size of transitions in the buffer.
        env_spec (garage.envs.EnvSpec): Environment specification.
    """

    def __init__(self, replay_k, reward_fun, capacity_in_transitions,
                 env_spec):
        self._replay_k = replay_k
        self._reward_fun = reward_fun
        self._env_spec = env_spec

        if not float(replay_k).is_integer() or replay_k < 0:
            raise ValueError('replay_k must be an integer.')
        super().__init__(capacity_in_transitions)

    def _sample_her_goals(self, path, transition_idx):
        """Samples HER goals from the given path.

        Goals are randomly sampled starting from the index after
        transition_idx in the given path.

        Args:
            path (dict[string:ndarray]): A dict containing the transition
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
        return path['achieved_goals'][goal_indexes]

    def sample_transitions(self, batch_size):
        """Sample a batch of random transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict[string:ndarray]: Each key in the dictionary will have
                shape :math:`(N, S^*)`. Note that the observations and
                next_observations are flattened np.ndarrays and not dicts.
        """
        transitions = super().sample_transitions(batch_size)
        obses = []
        next_obses = []
        for idx, _ in enumerate(transitions['observations']):

            obses.append(
                dict(observation=transitions['observations'][idx],
                     achieved_goal=transitions['achieved_goals'][idx],
                     desired_goal=transitions['desired_goals'][idx]))
            next_obses.append(
                dict(observation=transitions['next_observations'][idx],
                     achieved_goal=transitions['next_achieved_goals'][idx],
                     desired_goal=transitions['next_desired_goals'][idx]))

        transitions[
            'observations'] = self._env_spec.observation_space.flatten_n(obses)
        transitions[
            'next_observations'] = self._env_spec.observation_space.flatten_n(
                next_obses)

        del transitions['achieved_goals']
        del transitions['desired_goals']
        del transitions['next_achieved_goals']
        del transitions['next_desired_goals']

        return transitions

    def add_path(self, path):
        """Adds a path to the replay buffer.

        For each transition in the given path except the last one,
        replay_k HER transitions will added to the buffer in addition
        to the one in the path. The last transition is added without
        sampling additional HER goals.

        Args:
            path(dict[string:np.ndarray]): Each key in the dict must map
                to a np.ndarray of shape :math:`(T, S^*)`.

        """
        obses = path['observations']
        obs = np.asarray([obs['observation'] for obs in obses])
        obs = self._env_spec.observation_space['observation'].flatten_n(obs)
        d_g = np.asarray([obs['desired_goal'] for obs in obses])
        d_g = self._env_spec.observation_space['desired_goal'].flatten_n(d_g)
        a_g = np.asarray([obs['achieved_goal'] for obs in obses])
        a_g = self._env_spec.observation_space['achieved_goal'].flatten_n(a_g)

        next_obses = path['next_observations']
        next_obs = np.asarray(
            [next_obs['observation'] for next_obs in next_obses])
        next_obs = self._env_spec.observation_space['observation'].flatten_n(
            next_obs)
        next_ag = np.asarray(
            [next_obs['achieved_goal'] for next_obs in next_obses])
        next_ag = self._env_spec.observation_space['achieved_goal'].flatten_n(
            next_ag)
        next_dg = np.asarray(
            [next_obs['desired_goal'] for next_obs in next_obses])
        next_dg = self._env_spec.observation_space['desired_goal'].flatten_n(
            next_dg)

        actions = self._env_spec.action_space.flatten_n(path['actions'])

        path_dict = dict(observations=obs,
                         desired_goals=d_g,
                         rewards=path['rewards'],
                         achieved_goals=a_g,
                         terminals=path['terminals'],
                         actions=actions,
                         next_observations=next_obs,
                         next_achieved_goals=next_ag,
                         next_desired_goals=next_dg)

        super().add_path(path_dict)

        # create HER transitions and add them to the buffer

        for idx in range(actions.shape[0] - 1):
            transition = {
                key: sample[idx]
                for key, sample in path_dict.items()
            }
            her_goals = self._sample_her_goals(path_dict, idx)

            # create replay_k transitions using the HER goals
            for goal in her_goals:

                t_new = copy.deepcopy(transition)
                t_new['desired_goals'] = np.array(goal)
                t_new['next_desired_goals'] = np.array(goal)
                t_new['terminals'] = np.array(False)

                t_new['rewards'] = np.array(
                    self._reward_fun(t_new['next_achieved_goals'], goal, None))

                for key in t_new.keys():
                    t_new[key] = t_new[key].reshape(1, -1)

                # Since we're using a PathBuffer, add each transition
                # as its own path.
                super().add_path(t_new)

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
