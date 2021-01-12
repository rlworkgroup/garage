"""Data types for agent-based learning."""
from dataclasses import dataclass
import enum
from typing import Dict, List
import warnings

import numpy as np

from garage.np import (concat_tensor_dict_list, pad_batch_array,
                       slice_nested_dict, stack_tensor_dict_list)

# pylint: disable=too-many-lines


class StepType(enum.IntEnum):
    """Defines the status of a :class:`~TimeStep` within a sequence.

    Note that the last :class:`~TimeStep` in a sequence can either be
    :attribute:`StepType.TERMINAL` or :attribute:`StepType.TIMEOUT`.

    Suppose max_episode_length = 5:
    * A success sequence terminated at step 4 will look like:
        FIRST, MID, MID, TERMINAL
    * A success sequence terminated at step 5 will look like:
        FIRST, MID, MID, MID, TERMINAL
    * An unsuccessful sequence truncated by time limit will look like:
        FIRST, MID, MID, MID, TIMEOUT
    """
    # Denotes the first :class:`~TimeStep` in a sequence.
    FIRST = 0
    # Denotes any :class:`~TimeStep` in the middle of a sequence (i.e. not the
    # first or last one).
    MID = 1
    # Denotes the last :class:`~TimeStep` in a sequence that terminates
    # successfully.
    TERMINAL = 2
    # Denotes the last :class:`~TimeStep` in a sequence truncated by time
    # limit.
    TIMEOUT = 3

    @classmethod
    def get_step_type(cls, step_cnt, max_episode_length, done):
        """Determines the step type based on step cnt and done signal.

        Args:
            step_cnt (int): current step cnt of the environment.
            max_episode_length (int): maximum episode length.
            done (bool): the done signal returned by Environment.

        Returns:
            StepType: the step type.

        Raises:
            ValueError: if step_cnt is < 1. In this case a environment's
            `reset()` is likely not called yet and the step_cnt is None.
        """
        if max_episode_length is not None and step_cnt >= max_episode_length:
            return StepType.TIMEOUT
        elif done:
            return StepType.TERMINAL
        elif step_cnt == 1:
            return StepType.FIRST
        elif step_cnt < 1:
            raise ValueError('Expect step_cnt to be >= 1, but got {} '
                             'instead. Did you forget to call `reset('
                             ')`?'.format(step_cnt))
        else:
            return StepType.MID


@dataclass(frozen=True)
class TimeStep:
    r"""A single TimeStep in an environment.

    A :class:`~TimeStep` represents a single sample when an agent interacts
        with an environment. It describes as SARS (State–action–reward–state)
        tuple that characterizes the evolution of a MDP.

    Attributes:
        env_spec (EnvSpec): Specification for the environment from which this
            data was sampled.
        episode_info (dict[str, np.ndarray]): A dict of numpy arrays of shape
            :math:`(S*^,)` containing episode-level information of each
            episode.  For example, in goal-conditioned reinforcement learning
            this could contain the goal state for each episode.
        observation (numpy.ndarray): A numpy array of shape :math:`(O^*)`
            containing the observation for this time step in the
            environment. These must conform to
            :obj:`EnvStep.observation_space`.
            The observation before applying the action.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        action (numpy.ndarray): A numpy array of shape :math:`(A^*)`
            containing the action for this time step. These must conform
            to :obj:`EnvStep.action_space`.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        reward (float): A float representing the reward for taking the action
            given the observation, at this time step.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        next_observation (numpy.ndarray): A numpy array of shape :math:`(O^*)`
            containing the observation for this time step in the
            environment. These must conform to
            :obj:`EnvStep.observation_space`.
            The observation after applying the action.
        env_info (dict): A dict arbitrary environment state information.
        agent_info (dict): A dict of arbitrary agent
            state information. For example, this may contain the hidden states
            from an RNN policy.
        step_type (StepType): a :class:`~StepType` enum value. Can be one of
            :attribute:`~StepType.FIRST`, :attribute:`~StepType.MID`,
            :attribute:`~StepType.TERMINAL`, or :attribute:`~StepType.TIMEOUT`.

    """

    env_spec: 'garage.EnvSpec'  # NOQA: F821
    episode_info: Dict[str, np.ndarray]
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    env_info: Dict[str, np.ndarray]
    agent_info: Dict[str, np.ndarray]
    step_type: StepType

    @property
    def first(self):
        """bool: Whether this step is the first of its episode."""
        return self.step_type is StepType.FIRST

    @property
    def mid(self):
        """bool: Whether this step is in the middle of its episode."""
        return self.step_type is StepType.MID

    @property
    def terminal(self):
        """bool: Whether this step records a termination condition."""
        return self.step_type is StepType.TERMINAL

    @property
    def timeout(self):
        """bool: Whether this step records a timeout condition."""
        return self.step_type is StepType.TIMEOUT

    @property
    def last(self):
        """bool: Whether this step is the last of its episode."""
        return self.step_type is StepType.TERMINAL or self.step_type \
            is StepType.TIMEOUT

    @classmethod
    def from_env_step(cls, env_step, last_observation, agent_info,
                      episode_info):
        """Create a TimeStep from a EnvStep.

        Args:
            env_step (EnvStep): the env step returned by the environment.
            last_observation (numpy.ndarray): A numpy array of shape
                :math:`(O^*)` containing the observation for this time
                step in the environment. These must conform to
                :attr:`EnvStep.observation_space`.
                The observation before applying the action.
            agent_info (dict):  A dict of arbitrary agent state information.
            episode_info (dict):  A dict of arbitrary information associated
                with the whole episode.

        Returns:
            TimeStep: The TimeStep with all information of EnvStep plus the
            agent info.
        """
        return cls(env_spec=env_step.env_spec,
                   episode_info=episode_info,
                   observation=last_observation,
                   action=env_step.action,
                   reward=env_step.reward,
                   next_observation=env_step.observation,
                   env_info=env_step.env_info,
                   agent_info=agent_info,
                   step_type=env_step.step_type)


@dataclass(frozen=True)
class TimeStepBatch:
    # pylint: disable=missing-param-doc, missing-type-doc
    """A tuple representing a batch of TimeSteps.

    Data type for off-policy algorithms, imitation learning and batch-RL.

    Attributes:
        env_spec (EnvSpec): Specification for the environment from
            which this data was sampled.
        episode_infos (dict[str, np.ndarray]): A dict of numpy arrays
            containing the episode-level information of each episode. Each
            value of this dict should be a numpy array of shape :math:`(N,
            S^*)`. For example, in goal-conditioned reinforcement learning this
            could contain the goal state for each episode.
        observations (numpy.ndarray): Non-flattened array of observations.
            Typically has shape (batch_size, S^*) (the unflattened state space
            of the current environment).
        actions (numpy.ndarray): Non-flattened array of actions. Must
            have shape (batch_size, S^*) (the unflattened action space of the
            current environment).
        rewards (numpy.ndarray): Array of rewards of shape (batch_size, 1).
        next_observation (numpy.ndarray): Non-flattened array of next
            observations. Has shape (batch_size, S^*). next_observations[i] was
            observed by the agent after taking actions[i].
        env_infos (dict): A dict arbitrary environment state
            information.
        agent_infos (dict): A dict of arbitrary agent state information. For
            example, this may contain the hidden states from an RNN policy.
        step_types (numpy.ndarray): A numpy array of `StepType with shape (
            batch_size,) containing the time step types for all transitions in
            this batch.

    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """

    def __post_init__(self):
        """Runs integrity checking after __init__."""
        check_timestep_batch(self, np.ndarray)

    env_spec: 'garage.EnvSpec'  # NOQA: F821
    episode_infos: Dict[str, np.ndarray or dict]
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    agent_infos: Dict[str, np.ndarray or dict]
    env_infos: Dict[str, np.ndarray or dict]
    step_types: np.ndarray

    @classmethod
    def concatenate(cls, *batches):
        """Concatenate two or more :class:`TimeStepBatch`s.

        Args:
            batches (list[TimeStepBatch]): Batches to concatenate.

        Returns:
            TimeStepBatch: The concatenation of the batches.

        Raises:
            ValueError: If no TimeStepBatches are provided.

        """
        if len(batches) < 1:
            raise ValueError('Please provide at least one TimeStepBatch to '
                             'concatenate')
        episode_infos = {
            k: np.concatenate([b.episode_infos[k] for b in batches])
            for k in batches[0].episode_infos.keys()
        }
        env_infos = {
            k: np.concatenate([b.env_infos[k] for b in batches])
            for k in batches[0].env_infos.keys()
        }
        agent_infos = {
            k: np.concatenate([b.agent_infos[k] for b in batches])
            for k in batches[0].agent_infos.keys()
        }

        return cls(
            env_spec=batches[0].env_spec,
            episode_infos=episode_infos,
            observations=np.concatenate(
                [batch.observations for batch in batches]),
            actions=np.concatenate([batch.actions for batch in batches]),
            rewards=np.concatenate([batch.rewards for batch in batches]),
            next_observations=np.concatenate(
                [batch.next_observations for batch in batches]),
            env_infos=env_infos,
            agent_infos=agent_infos,
            step_types=np.concatenate([batch.step_types for batch in batches]))

    def split(self) -> List['TimeStepBatch']:
        """Split a :class:`~TimeStepBatch` into a list of :class:`~TimeStepBatch`s.

        The opposite of concatenate.

        Returns:
            list[TimeStepBatch]: A list of :class:`TimeStepBatch`s, with one
                :class:`~TimeStep` per :class:`~TimeStepBatch`.

        """
        time_steps = []

        for i in range(len(self.rewards)):
            time_step = TimeStepBatch(
                episode_infos={
                    k: np.asarray([v[i]])
                    for (k, v) in self.episode_infos.items()
                },
                env_spec=self.env_spec,
                observations=np.asarray([self.observations[i]]),
                actions=np.asarray([self.actions[i]]),
                rewards=np.asarray([self.rewards[i]]),
                next_observations=np.asarray([self.next_observations[i]]),
                env_infos={
                    k: np.asarray([v[i]])
                    for (k, v) in self.env_infos.items()
                },
                agent_infos={
                    k: np.asarray([v[i]])
                    for (k, v) in self.agent_infos.items()
                },
                step_types=np.asarray([self.step_types[i]], dtype=StepType))
            time_steps.append(time_step)
        return time_steps

    def to_time_step_list(self) -> List[Dict[str, np.ndarray]]:
        """Convert the batch into a list of dictionaries.

        Breaks the :class:`~TimeStepBatch` into a list of single time step
        sample dictionaries. len(rewards) (or the number of discrete time step)
        dictionaries are returned

        Returns:
            list[dict[str, np.ndarray or dict[str, np.ndarray]]]: Keys:
                episode_infos (dict[str, np.ndarray]): A dict of numpy arrays
                    containing the episode-level information of each episode.
                    Each value of this dict must be a numpy array of shape
                    :math:`(S^*,)`. For example, in goal-conditioned
                    reinforcement learning this could contain the goal state
                    for each episode.
                observations (numpy.ndarray): Non-flattened array of
                    observations.
                    Typically has shape (batch_size, S^*) (the unflattened
                    state space
                    of the current environment).
                actions (numpy.ndarray): Non-flattened array of actions. Must
                    have shape (batch_size, S^*) (the unflattened action
                    space of the
                    current environment).
                rewards (numpy.ndarray): Array of rewards of shape (
                    batch_size,) (1D array of length batch_size).
                next_observation (numpy.ndarray): Non-flattened array of next
                    observations. Has shape (batch_size, S^*).
                    next_observations[i] was
                    observed by the agent after taking actions[i].
                env_infos (dict): A dict arbitrary environment state
                    information.
                agent_infos (dict): A dict of arbitrary agent state
                    information. For example, this may contain the
                    hidden states from an RNN policy.
                step_types (numpy.ndarray): A numpy array of `StepType with
                        shape (batch_size,) containing the time step types for
                        all transitions in this batch.
        """
        samples = []
        for i in range(len(self.rewards)):
            samples.append({
                'episode_infos': {
                    k: np.asarray([v[i]])
                    for (k, v) in self.episode_infos.items()
                },
                'observations':
                np.asarray([self.observations[i]]),
                'actions':
                np.asarray([self.actions[i]]),
                'rewards':
                np.asarray([self.rewards[i]]),
                'next_observations':
                np.asarray([self.next_observations[i]]),
                'env_infos':
                {k: np.asarray([v[i]])
                 for (k, v) in self.env_infos.items()},
                'agent_infos':
                {k: np.asarray([v[i]])
                 for (k, v) in self.agent_infos.items()},
                'step_types':
                np.asarray([self.step_types[i]])
            })
        return samples

    @property
    def terminals(self):
        """Get an array of boolean indicating ternianal information.

        Returns:
            numpy.ndarray: An array of boolean of shape :math:`(N,)`
                indicating whether the `StepType is `TERMINAL

        """
        return np.array([s == StepType.TERMINAL for s in self.step_types])

    @classmethod
    def from_time_step_list(cls, env_spec, ts_samples):
        """Create a :class:`~TimeStepBatch` from a list of time step dictionaries.

        Args:
            env_spec (EnvSpec): Specification for the environment from which
                this data was sampled.
            ts_samples (list[dict[str, np.ndarray or dict[str, np.ndarray]]]):
                keys:
                * episode_infos (dict[str, np.ndarray]): A dict of numpy arrays
                    containing the episode-level information of each episode.
                    Each value of this dict must be a numpy array of shape
                    :math:`(N, S^*)`. For example, in goal-conditioned
                    reinforcement learning this could contain the goal state
                    for each episode.
                * observations (numpy.ndarray): Non-flattened array of
                    observations.
                    Typically has shape (batch_size, S^*) (the unflattened
                    state space of the current environment).
                * actions (numpy.ndarray): Non-flattened array of actions.
                    Must have shape (batch_size, S^*) (the unflattened action
                    space of the current environment).
                * rewards (numpy.ndarray): Array of rewards of shape (
                    batch_size,) (1D array of length batch_size).
                * next_observation (numpy.ndarray): Non-flattened array of next
                    observations. Has shape (batch_size, S^*).
                    next_observations[i] was observed by the agent after
                    taking actions[i].
                * env_infos (dict): A dict arbitrary environment state
                    information.
                * agent_infos (dict): A dict of arbitrary agent
                    state information. For example, this may contain the
                    hidden states from an RNN policy.
                * step_types (numpy.ndarray): A numpy array of `StepType with
                shape (batch_size,) containing the time step types for all
                    transitions in this batch.

        Returns:
            TimeStepBatch: The concatenation of samples.

        Raises:
            ValueError: If no dicts are provided.

        """
        if len(ts_samples) < 1:
            raise ValueError('Please provide at least one dict')

        ts_batches = [
            TimeStepBatch(episode_infos=sample['episode_infos'],
                          env_spec=env_spec,
                          observations=sample['observations'],
                          actions=sample['actions'],
                          rewards=sample['rewards'],
                          next_observations=sample['next_observations'],
                          env_infos=sample['env_infos'],
                          agent_infos=sample['agent_infos'],
                          step_types=sample['step_types'])
            for sample in ts_samples
        ]

        return TimeStepBatch.concatenate(*ts_batches)


@dataclass(frozen=True, init=False)
class EpisodeBatch(TimeStepBatch):
    # pylint: disable=missing-return-doc, missing-return-type-doc, missing-param-doc, missing-type-doc  # noqa: E501
    r"""A tuple representing a batch of whole episodes.

    Data type for on-policy algorithms.

    A :class:`~EpisodeBatch` represents a batch of whole episodes, produced
    when one or more agents interacts with one or more environments.

    +-----------------------+-------------------------------------------------+
    | Symbol                | Description                                     |
    +=======================+=================================================+
    | :math:`N`             | Episode batch dimension                         |
    +-----------------------+-------------------------------------------------+
    | :math:`[T]`           | Variable-length time dimension of each          |
    |                       | episode                                         |
    +-----------------------+-------------------------------------------------+
    | :math:`S^*`           | Single-step shape of a time-series tensor       |
    +-----------------------+-------------------------------------------------+
    | :math:`N \bullet [T]` | A dimension computed by flattening a            |
    |                       | variable-length time dimension :math:`[T]` into |
    |                       | a single batch dimension with length            |
    |                       | :math:`sum_{i \in N} [T]_i`                     |
    +-----------------------+-------------------------------------------------+

    Attributes:
        env_spec (EnvSpec): Specification for the environment from
            which this data was sampled.
        episode_infos (dict[str, np.ndarray]): A dict of numpy arrays
            containing the episode-level information of each episode. Each
            value of this dict should be a numpy array of shape :math:`(N,
            S^*)`. For example, in goal-conditioned reinforcement learning this
            could contain the goal state for each episode.
        observations (numpy.ndarray): A numpy array of shape
            :math:`(N \bullet [T], O^*)` containing the (possibly
            multi-dimensional) observations for all time steps in this batch.
            These must conform to :obj:`EnvStep.observation_space`.
        last_observations (numpy.ndarray): A numpy array of shape
            :math:`(N, O^*)` containing the last observation of each episode.
            This is necessary since there are one more observations than
            actions every episode.
        actions (numpy.ndarray): A  numpy array of shape
            :math:`(N \bullet [T], A^*)` containing the (possibly
            multi-dimensional) actions for all time steps in this batch. These
            must conform to :obj:`EnvStep.action_space`.
        rewards (numpy.ndarray): A numpy array of shape
            :math:`(N \bullet [T])` containing the rewards for all time steps
            in this batch.
        env_infos (dict[str, np.ndarray]): A dict of numpy arrays arbitrary
            environment state information. Each value of this dict should be
            a numpy array of shape :math:`(N \bullet [T])` or :math:`(N \bullet
            [T], S^*)`.
        agent_infos (dict[str, np.ndarray]): A dict of numpy arrays arbitrary
            agent state information. Each value of this dict should be a numpy
            array of shape :math:`(N \bullet [T])` or :math:`(N \bullet [T],
            S^*)`.  For example, this may contain the hidden states from an RNN
            policy.
        step_types (numpy.ndarray): A numpy array of `StepType with shape
            :math:`(N \bullet [T])` containing the time step types for all
            transitions in this batch.
        lengths (numpy.ndarray): An integer numpy array of shape :math:`(N,)`
            containing the length of each episode in this batch. This may be
            used to reconstruct the individual episodes.

    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """
    episode_infos_by_episode: np.ndarray
    last_observations: np.ndarray
    lengths: np.ndarray

    def __init__(self, env_spec, episode_infos, observations,
                 last_observations, actions, rewards, env_infos, agent_infos,
                 step_types, lengths):  # noqa: D102
        # lengths
        if len(lengths.shape) != 1:
            raise ValueError(
                f'lengths has shape {lengths.shape} but must be a ternsor of '
                f'shape (N,)')

        if not (lengths.dtype.kind == 'u' or lengths.dtype.kind == 'i'):
            raise ValueError(
                f'lengths has dtype {lengths.dtype}, but must have an '
                f'integer dtype')

        n_episodes = len(lengths)

        # Check episode_infos and last_observations here instead of checking
        # episode_infos and next_observations in check_timestep_batch.

        for key, val in episode_infos.items():
            if not isinstance(val, np.ndarray):
                raise ValueError(
                    f'Entry {key!r} in episode_infos is of type {type(val)!r} '
                    f'but must be of type {np.ndarray!r}')
            if hasattr(val, 'shape'):
                if val.shape[0] != n_episodes:
                    raise ValueError(
                        f'Entry {key!r} in episode_infos has batch size '
                        f'{val.shape[0]}, but must have batch size '
                        f'{n_episodes} to match the number of episodes')

        if not isinstance(last_observations, np.ndarray):
            raise ValueError(
                f'last_observations is not of type {np.ndarray!r}')
        if last_observations.shape[0] != n_episodes:
            raise ValueError(
                f'last_observations has batch size '
                f'{last_observations.shape[0]} but must have '
                f'batch size {n_episodes} to match the number of episodes')
        if not _space_soft_contains(env_spec.observation_space,
                                    last_observations[0]):
            raise ValueError(f'last_observations must have the same '
                             f'number of entries as there are episodes '
                             f'({n_episodes}) but got data with shape '
                             '{last_observations[0].shape} entries')

        object.__setattr__(self, 'last_observations', last_observations)
        object.__setattr__(self, 'lengths', lengths)
        object.__setattr__(self, 'env_spec', env_spec)
        # Used to compute the episode_infos property, but also used in .split
        object.__setattr__(self, 'episode_infos_by_episode', episode_infos)
        object.__setattr__(self, 'observations', observations)
        # No need for next_observations, it was replaced with a property
        object.__setattr__(self, 'actions', actions)
        object.__setattr__(self, 'rewards', rewards)
        object.__setattr__(self, 'env_infos', env_infos)
        object.__setattr__(self, 'agent_infos', agent_infos)
        object.__setattr__(self, 'step_types', step_types)
        check_timestep_batch(
            self,
            np.ndarray,
            ignored_fields={'next_observations', 'episode_infos'})

    @classmethod
    def concatenate(cls, *batches):
        """Create a EpisodeBatch by concatenating EpisodeBatches.

        Args:
            batches (list[EpisodeBatch]): Batches to concatenate.

        Returns:
            EpisodeBatch: The concatenation of the batches.

        """
        if __debug__:
            for b in batches:
                assert (set(b.env_infos.keys()) == set(
                    batches[0].env_infos.keys()))
                assert (set(b.agent_infos.keys()) == set(
                    batches[0].agent_infos.keys()))
        env_infos = {
            k: np.concatenate([b.env_infos[k] for b in batches])
            for k in batches[0].env_infos.keys()
        }
        agent_infos = {
            k: np.concatenate([b.agent_infos[k] for b in batches])
            for k in batches[0].agent_infos.keys()
        }
        episode_infos = {
            k: np.concatenate([b.episode_infos_by_episode[k] for b in batches])
            for k in batches[0].episode_infos_by_episode.keys()
        }
        return cls(
            episode_infos=episode_infos,
            env_spec=batches[0].env_spec,
            observations=np.concatenate(
                [batch.observations for batch in batches]),
            last_observations=np.concatenate(
                [batch.last_observations for batch in batches]),
            actions=np.concatenate([batch.actions for batch in batches]),
            rewards=np.concatenate([batch.rewards for batch in batches]),
            env_infos=env_infos,
            agent_infos=agent_infos,
            step_types=np.concatenate([batch.step_types for batch in batches]),
            lengths=np.concatenate([batch.lengths for batch in batches]))

    def _episode_ranges(self):
        """Iterate through start and stop indices for each episode.

        Yields:
            tuple[int, int]: Start index (inclusive) and stop index
                (exclusive).

        """
        start = 0
        for length in self.lengths:
            stop = start + length
            yield (start, stop)
            start = stop

    def split(self):
        """Split an EpisodeBatch into a list of EpisodeBatches.

        The opposite of concatenate.

        Returns:
            list[EpisodeBatch]: A list of EpisodeBatches, with one
                episode per batch.

        """
        episodes = []
        for i, (start, stop) in enumerate(self._episode_ranges()):
            eps = EpisodeBatch(
                env_spec=self.env_spec,
                episode_infos=slice_nested_dict(self.episode_infos_by_episode,
                                                i, i + 1),
                observations=self.observations[start:stop],
                last_observations=np.asarray([self.last_observations[i]]),
                actions=self.actions[start:stop],
                rewards=self.rewards[start:stop],
                env_infos=slice_nested_dict(self.env_infos, start, stop),
                agent_infos=slice_nested_dict(self.agent_infos, start, stop),
                step_types=self.step_types[start:stop],
                lengths=np.asarray([self.lengths[i]]))
            episodes.append(eps)

        return episodes

    def to_list(self):
        """Convert the batch into a list of dictionaries.

        Returns:
            list[dict[str, np.ndarray or dict[str, np.ndarray]]]: Keys:
                * observations (np.ndarray): Non-flattened array of
                    observations. Has shape (T, S^*) (the unflattened state
                    space of the current environment).  observations[i] was
                    used by the agent to choose actions[i].
                * next_observations (np.ndarray): Non-flattened array of
                    observations. Has shape (T, S^*). next_observations[i] was
                    observed by the agent after taking actions[i].
                * actions (np.ndarray): Non-flattened array of actions. Must
                    have shape (T, S^*) (the unflattened action space of the
                    current environment).
                * rewards (np.ndarray): Array of rewards of shape (T,) (1D
                    array of length timesteps).
                * agent_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `agent_info` arrays.
                * env_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `env_info` arrays.
                * step_types (numpy.ndarray): A numpy array of `StepType with
                    shape (T,) containing the time step types for all
                    transitions in this batch.
                * episode_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `episode_info` arrays.

        """
        episodes = []
        for i, (start, stop) in enumerate(self._episode_ranges()):
            episodes.append({
                'episode_infos':
                {k: v[i:i + 1]
                 for (k, v) in self.episode_infos.items()},
                'observations':
                self.observations[start:stop],
                'next_observations':
                np.concatenate((self.observations[1 + start:stop],
                                [self.last_observations[i]])),
                'actions':
                self.actions[start:stop],
                'rewards':
                self.rewards[start:stop],
                'env_infos':
                {k: v[start:stop]
                 for (k, v) in self.env_infos.items()},
                'agent_infos':
                {k: v[start:stop]
                 for (k, v) in self.agent_infos.items()},
                'step_types':
                self.step_types[start:stop]
            })
        return episodes

    @classmethod
    def from_list(cls, env_spec, paths):
        """Create a EpisodeBatch from a list of episodes.

        Args:
            env_spec (EnvSpec): Specification for the environment from which
                this data was sampled.
            paths (list[dict[str, np.ndarray or dict[str, np.ndarray]]]): Keys:
                * episode_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `episode_info` arrays, each of shape (S^*).
                * observations (np.ndarray): Non-flattened array of
                    observations. Typically has shape (T, S^*) (the unflattened
                    state space of the current environment). observations[i]
                    was used by the agent to choose actions[i]. observations
                    may instead have shape (T + 1, S^*).
                * next_observations (np.ndarray): Non-flattened array of
                    observations. Has shape (T, S^*). next_observations[i] was
                    observed by the agent after taking actions[i]. Optional.
                    Note that to ensure all information from the environment
                    was preserved, observations[i] must have shape (T + 1,
                    S^*), or this key must be set. However, this method is
                    lenient and will "duplicate" the last observation if the
                    original last observation has been lost.
                * actions (np.ndarray): Non-flattened array of actions. Must
                    have shape (T, S^*) (the unflattened action space of the
                    current environment).
                * rewards (np.ndarray): Array of rewards of shape (T,) (1D
                    array of length timesteps).
                * agent_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `agent_info` arrays.
                * env_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `env_info` arrays.
                * step_types (numpy.ndarray): A numpy array of `StepType with
                    shape (T,) containing the time step types for all
                    transitions in this batch.
        """
        lengths = np.asarray([len(p['rewards']) for p in paths])
        if all(
                len(path['observations']) == length + 1
                for (path, length) in zip(paths, lengths)):
            last_observations = np.asarray(
                [p['observations'][-1] for p in paths])
            observations = np.concatenate(
                [p['observations'][:-1] for p in paths])
        else:
            # The number of observations and timesteps must match.
            observations = np.concatenate([p['observations'] for p in paths])
            if paths[0].get('next_observations') is not None:
                last_observations = np.asarray(
                    [p['next_observations'][-1] for p in paths])
            else:
                last_observations = np.asarray(
                    [p['observations'][-1] for p in paths])

        stacked_paths = concat_tensor_dict_list(paths)
        episode_infos = stack_tensor_dict_list(
            [path['episode_infos'] for path in paths])

        # Temporary solution. This logic is not needed if algorithms process
        # step_types instead of dones directly.
        if 'dones' in stacked_paths and 'step_types' not in stacked_paths:
            step_types = np.array([
                StepType.TERMINAL if done else StepType.MID
                for done in stacked_paths['dones']
            ],
                                  dtype=StepType)
            stacked_paths['step_types'] = step_types
            del stacked_paths['dones']

        return cls(env_spec=env_spec,
                   episode_infos=episode_infos,
                   observations=observations,
                   last_observations=last_observations,
                   actions=stacked_paths['actions'],
                   rewards=stacked_paths['rewards'],
                   env_infos=stacked_paths['env_infos'],
                   agent_infos=stacked_paths['agent_infos'],
                   step_types=stacked_paths['step_types'],
                   lengths=lengths)

    @property
    def next_observations(self):
        r"""Get the observations seen after actions are performed.

        In an :class:`~EpisodeBatch`, next_observations don't need to be stored
        explicitly, since the next observation is already stored in
        the batch.

        Returns:
            np.ndarray: The "next_observations" with shape
                :math:`(N \bullet [T], O^*)`

        """
        return np.concatenate(
            tuple([
                np.concatenate((eps.observations[1:], eps.last_observations))
                for eps in self.split()
            ]))

    @property
    def episode_infos(self):
        r"""Get the episode_infos.

        In an :class:`~EpisodeBatch`, episode_infos only need to be stored once
        per episode. However, the episode_infos field of
        :class:`~TimeStepBatch` has shape :math:`(N \bullet [T])`. This method
        expands episode_infos_by_episode (which have shape :math:`(N)`) to
        :math:`(N \bullet [T])`.

        Returns:
            dict[str, np.ndarray]: The episode_infos each of length :math:`(N
                \bullet [T])`.

        """
        return {
            key: np.concatenate([
                np.repeat([v], length, axis=0)
                for (v, length) in zip(val, self.lengths)
            ])
            for (key, val) in self.episode_infos_by_episode.items()
        }

    @property
    def padded_observations(self):
        """Padded observations.

        Returns:
            np.ndarray: Padded observations with shape of
                :math:`(N, max_episode_length, O^*)`.

        """
        return pad_batch_array(self.observations, self.lengths,
                               self.env_spec.max_episode_length)

    @property
    def padded_actions(self):
        """Padded actions.

        Returns:
            np.ndarray: Padded actions with shape of
                :math:`(N, max_episode_length, A^*)`.

        """
        return pad_batch_array(self.actions, self.lengths,
                               self.env_spec.max_episode_length)

    @property
    def observations_list(self):
        """Split observations into a list.

        Returns:
            list[np.ndarray]: Splitted list.

        """
        obs_list = []
        for start, stop in self._episode_ranges():
            obs_list.append(self.observations[start:stop])
        return obs_list

    @property
    def actions_list(self):
        """Split actions into a list.

        Returns:
            list[np.ndarray]: Splitted list.

        """
        acts_list = []
        for start, stop in self._episode_ranges():
            acts_list.append(self.actions[start:stop])
        return acts_list

    @property
    def padded_rewards(self):
        """Padded rewards.

        Returns:
            np.ndarray: Padded rewards with shape of
                :math:`(N, max_episode_length)`.

        """
        return pad_batch_array(self.rewards, self.lengths,
                               self.env_spec.max_episode_length)

    @property
    def valids(self):
        """An array indicating valid steps in a padded tensor.

        Returns:
            np.ndarray: the shape is :math:`(N, max_episode_length)`.

        """
        return pad_batch_array(np.ones_like(self.rewards), self.lengths,
                               self.env_spec.max_episode_length)

    @property
    def padded_next_observations(self):
        """Padded next_observations array.

        Returns:
            np.ndarray: Array of shape :math:`(N, max_episode_length, O^*)`

        """
        return pad_batch_array(self.next_observations, self.lengths,
                               self.env_spec.max_episode_length)

    @property
    def padded_step_types(self):
        """Padded step_type array.

        Returns:
            np.ndarray: Array of shape :math:`(N, max_episode_length)`

        """
        return pad_batch_array(self.step_types, self.lengths,
                               self.env_spec.max_episode_length)

    @property
    def padded_agent_infos(self):
        """Padded agent infos.

        Returns:
            dict[str, np.ndarray]: Padded agent infos. Each value must have
                shape with :math:`(N, max_episode_length)` or
                :math:`(N, max_episode_length, S^*)`.

        """
        return {
            k: pad_batch_array(arr, self.lengths,
                               self.env_spec.max_episode_length)
            for (k, arr) in self.agent_infos.items()
        }

    @property
    def padded_env_infos(self):
        """Padded env infos.

        Returns:
            dict[str, np.ndarray]: Padded env infos. Each value must have
                shape with :math:`(N, max_episode_length)` or
                :math:`(N, max_episode_length, S^*)`.

        """
        return {
            k: pad_batch_array(arr, self.lengths,
                               self.env_spec.max_episode_length)
            for (k, arr) in self.env_infos.items()
        }


def _space_soft_contains(space, element):
    """Check that a space has the same dimensionality as an element.

    If the space's dimensionality is not available, check that the space
    contains the element.

    Args:
        space (akro.Space or gym.Space): Space to check
        element (object): Element to check in space.

    Returns:
        bool: True iff the element was "matched" the space.
    """
    if space.contains(element):
        return True
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim == np.prod(element.shape)
    else:
        return False


def check_timestep_batch(batch, array_type, ignored_fields=()):
    """Check a TimeStepBatch of any array type that has .shape.

    Args:
        batch (TimeStepBatch): Batch of timesteps.
        array_type (type): Array type.
        ignored_fields (set[str]): Set of fields to ignore checking on.

    Raises:
        ValueError: If an invariant of TimeStepBatch is broken.
    """
    # pylint:disable=too-many-branches
    fields = {
        field: getattr(batch, field)
        for field in [
            'env_spec', 'rewards', 'rewards', 'observations', 'actions',
            'next_observations', 'step_types', 'agent_infos', 'episode_infos',
            'env_infos'
        ] if field not in ignored_fields
    }
    env_spec = fields.get('env_spec', None)
    inferred_batch_size = None
    inferred_batch_size_field = None
    for field, value in fields.items():
        if field in [
                'observations', 'actions', 'rewards', 'next_observations',
                'step_types'
        ]:
            if not isinstance(value, array_type):
                raise ValueError(f'{field} is not of type {array_type!r}')
        if hasattr(value, 'shape'):
            if inferred_batch_size is None:
                inferred_batch_size = value.shape[0]
                inferred_batch_size_field = field
            elif value.shape[0] != inferred_batch_size:
                raise ValueError(
                    f'{field} has batch size {value.shape[0]}, but '
                    f'must have batch size {inferred_batch_size} '
                    f'to match {inferred_batch_size_field}')
            if env_spec and field in ['observations', 'next_observations']:
                if not _space_soft_contains(env_spec.observation_space,
                                            value[0]):
                    raise ValueError(
                        f'Each {field[:-1]} has shape {value[0].shape} '
                        f'but must match the observation_space '
                        f'{env_spec.observation_space}')
                if (isinstance(value[0], np.ndarray)
                        and not env_spec.observation_space.contains(value[0])):
                    warnings.warn(
                        f'Observation {value[0]!r} is outside '
                        f'observation_space {env_spec.observation_space}')
            if env_spec and field == 'actions':
                if not _space_soft_contains(env_spec.action_space, value[0]):
                    raise ValueError(
                        f'Each {field[:-1]} has shape {value[0].shape} '
                        f'but must match the action_space '
                        f'{env_spec.action_space}')
            if field in ['rewards', 'step_types']:
                if value.shape != (inferred_batch_size, ):
                    raise ValueError(f'{field} has shape {value.shape} '
                                     f'but must have batch size '
                                     f'{inferred_batch_size} to match '
                                     f'{inferred_batch_size_field}')
        if field in ['agent_infos', 'env_infos', 'episode_infos']:
            for key, val in value.items():
                if not isinstance(val, (array_type, dict)):
                    raise ValueError(
                        f'Entry {key!r} in {field} is of type {type(val)}'
                        f'but must be {array_type!r} or dict')
                if hasattr(val, 'shape'):
                    if val.shape[0] != inferred_batch_size:
                        raise ValueError(
                            f'Entry {key!r} in {field} has batch size '
                            f'{val.shape[0]} but must have batch size '
                            f'{inferred_batch_size} to match '
                            f'{inferred_batch_size_field}')

        if (field == 'step_types' and isinstance(value, np.ndarray)
                and  # Only numpy arrays support custom dtypes.
                value.dtype != StepType):
            raise ValueError(
                f'step_types has dtype {value.dtype} but must have '
                f'dtype StepType')
