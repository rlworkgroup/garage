"""This module implements a Hindsight Experience Replay (HER).

See: https://arxiv.org/abs/1707.01495.
"""
import inspect

import numpy as np

from garage.replay_buffer.base import ReplayBuffer


def make_her_sample(replay_k, reward_fun):
    """Generate a transition sampler for HER ReplayBuffer.

    Args:
        replay_k (float): Ratio between HER replays and regular replays
        reward_fun (callable): Function to re-compute the reward with
            substituted goals

    Returns:
        callable: A function that returns sample transitions for HER.

    """
    future_p = 1 - (1. / (1 + replay_k))

    def _her_sample_transitions(episode_batch, sample_batch_size):
        """Generate a dictionary of transitions.

        Args:
            episode_batch (dict): Original transitions which
                transitions[key] has shape [batch_size, T, dim]
            sample_batch_size (int): Batch size per sample.

        Returns:
            dict[numpy.ndarray]: Transitions which transitions[key] has the
                shape of [sample_batch_size, dim]. Keys include `observation`,
                `action`, `goal`, `achieved_goal`, `terminal`,
                `next_observation`, `next_achieved_goal` and `reward`.

        """
        # Select which episodes to use
        time_horizon = episode_batch['action'].shape[1]
        rollout_batch_size = episode_batch['action'].shape[0]
        episode_idxs = np.random.randint(rollout_batch_size,
                                         size=sample_batch_size)
        # Select time steps to use
        t_samples = np.random.randint(time_horizon, size=sample_batch_size)
        transitions = {
            key: episode_batch[key][episode_idxs, t_samples]
            for key in episode_batch.keys()
        }

        her_idxs = np.where(
            np.random.uniform(size=sample_batch_size) < future_p)
        future_offset = np.random.uniform(
            size=sample_batch_size) * (time_horizon - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + future_offset)[her_idxs]

        future_ag = episode_batch['achieved_goal'][episode_idxs[her_idxs],
                                                   future_t]
        transitions['goal'][her_idxs] = future_ag

        # Re-compute reward since we may have substituted the goal.
        reward_params_keys = inspect.signature(reward_fun).parameters.keys()
        reward_params = {
            rk: transitions[k]
            for k, rk in zip(['next_achieved_goal', 'goal'],
                             list(reward_params_keys)[:-1])
        }
        reward_params['info'] = {}
        transitions['reward'] = reward_fun(**reward_params)

        transitions = {
            k: transitions[k].reshape(sample_batch_size,
                                      *transitions[k].shape[1:])
            for k in transitions.keys()
        }
        assert transitions['action'].shape[0] == sample_batch_size
        return transitions

    return _her_sample_transitions


class HerReplayBuffer(ReplayBuffer):
    """This class implements HerReplayBuffer.

    It constructs hindsight examples using future strategy.

    Args:
        replay_k (float): Ratio between HER replays and regular replays
        reward_fun (callable): Function to re-compute the reward with
            substituted goals
        kwargs: Keyword arguments for base class

    """

    def __init__(self, replay_k, reward_fun, **kwargs):
        self._sample_transitions = make_her_sample(replay_k, reward_fun)
        super(HerReplayBuffer, self).__init__(**kwargs)

    def sample(self, batch_size):
        """Sample a transition of batch_size.

        Args:
            batch_size (int): Batch size to sample.

        Return:
            dict[numpy.ndarray]: See _her_sample_transitions above.

        """
        buffer = {}
        for key in self._buffer:
            buffer[key] = self._buffer[key][:self._current_size]

        transitions = self._sample_transitions(buffer, batch_size)

        for key in (['reward', 'next_observation', 'next_achieved_goal'] +
                    list(self._buffer.keys())):
            assert key in transitions, 'key %s missing from transitions' % key

        return transitions

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_sample_transitions']
        return new_dict
