"""Data types for agent-based learning."""
import collections
import enum

import akro
import numpy as np

from garage.misc import tensor_utils

# pylint: disable=too-many-lines


class EpisodeBatch(
        collections.namedtuple('EpisodeBatch', [
            'env_spec',
            'observations',
            'last_observations',
            'actions',
            'rewards',
            'env_infos',
            'agent_infos',
            'step_types',
            'lengths',
        ])):
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
        env_infos (dict): A dict of numpy arrays arbitrary environment state
            information. Each value of this dict should be a numpy array of
            shape :math:`(N \bullet [T])` or :math:`(N \bullet [T], S^*)`.
        agent_infos (numpy.ndarray): A dict of numpy arrays arbitrary agent
            state information. Each value of this dict should be a numpy array
            of shape :math:`(N \bullet [T])` or :math:`(N \bullet [T], S^*)`.
            For example, this may contain the hidden states from an RNN policy.
        step_types (numpy.ndarray): A numpy array of `StepType with shape
            :math:`(N,)` containing the time step types for all transitions in
            this batch.
        lengths (numpy.ndarray): An integer numpy array of shape :math:`(N,)`
            containing the length of each episode in this batch. This may be
            used to reconstruct the individual episodes.

    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """
    __slots__ = ()

    def __new__(cls, env_spec, observations, last_observations, actions,
                rewards, env_infos, agent_infos, step_types,
                lengths):  # noqa: D102
        # pylint: disable=too-many-branches

        first_observation = observations[0]
        first_action = actions[0]
        inferred_batch_size = lengths.sum()

        # lengths
        if len(lengths.shape) != 1:
            raise ValueError(
                'Lengths tensor must be a tensor of shape (N,), but got a '
                'tensor of shape {} instead'.format(lengths.shape))

        if not (lengths.dtype.kind == 'u' or lengths.dtype.kind == 'i'):
            raise ValueError(
                'Lengths tensor must have an integer dtype, but got dtype {} '
                'instead.'.format(lengths.dtype))

        # observations
        if not env_spec.observation_space.contains(first_observation):
            # Discrete actions can be either in the space normally, or one-hot
            # encoded.
            if isinstance(env_spec.observation_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.observation_space.flat_dim != np.prod(
                        first_observation.shape):
                    raise ValueError('observations should have the same '
                                     'dimensionality as the observation_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.observation_space.flat_dim,
                                         first_observation.shape))
            else:
                raise ValueError(
                    'observations must conform to observation_space {}, but '
                    'got data with shape {} instead.'.format(
                        env_spec.observation_space, first_observation))

        if observations.shape[0] != inferred_batch_size:
            raise ValueError(
                'Expected batch dimension of observations to be length {}, '
                'but got length {} instead.'.format(inferred_batch_size,
                                                    observations.shape[0]))

        # observations
        if not env_spec.observation_space.contains(last_observations[0]):
            # Discrete actions can be either in the space normally, or one-hot
            # encoded.
            if isinstance(env_spec.observation_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.observation_space.flat_dim != np.prod(
                        last_observations[0].shape):
                    raise ValueError('last_observations should have the same '
                                     'dimensionality as the observation_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.observation_space.flat_dim,
                                         last_observations[0].shape))
            else:
                raise ValueError(
                    'last_observations must conform to observation_space {}, '
                    'but got data with shape {} instead.'.format(
                        env_spec.observation_space, last_observations[0]))

        if last_observations.shape[0] != len(lengths):
            raise ValueError(
                'Expected batch dimension of last_observations to be length '
                '{}, but got length {} instead.'.format(
                    len(lengths), last_observations.shape[0]))

        # actions
        if not env_spec.action_space.contains(first_action):
            # Discrete actions can be either in the space normally, or one-hot
            # encoded.
            if isinstance(env_spec.action_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.action_space.flat_dim != np.prod(
                        first_action.shape):
                    raise ValueError('actions should have the same '
                                     'dimensionality as the action_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.action_space.flat_dim,
                                         first_action.shape))
            else:
                raise ValueError(
                    'actions must conform to action_space {}, but got data '
                    'with shape {} instead.'.format(env_spec.action_space,
                                                    first_action))

        if actions.shape[0] != inferred_batch_size:
            raise ValueError(
                'Expected batch dimension of actions to be length {}, but got '
                'length {} instead.'.format(inferred_batch_size,
                                            actions.shape[0]))

        # rewards
        if rewards.shape != (inferred_batch_size, ):
            raise ValueError(
                'Rewards tensor must have shape {}, but got shape {} '
                'instead.'.format(inferred_batch_size, rewards.shape))

        # env_infos
        for key, val in env_infos.items():
            if not isinstance(val, (dict, np.ndarray)):
                raise ValueError(
                    'Each entry in env_infos must be a numpy array or '
                    'dictionary, but got key {} with value type {} instead.'.
                    format(key, type(val)))

            if (isinstance(val, np.ndarray)
                    and val.shape[0] != inferred_batch_size):
                raise ValueError(
                    'Each entry in env_infos must have a batch dimension of '
                    'length {}, but got key {} with batch size {} instead.'.
                    format(inferred_batch_size, key, val.shape[0]))

        # agent_infos
        for key, val in agent_infos.items():
            if not isinstance(val, (dict, np.ndarray)):
                raise ValueError(
                    'Each entry in agent_infos must be a numpy array or '
                    'dictionary, but got key {} with value type {} instead.'
                    'instead'.format(key, type(val)))

            if (isinstance(val, np.ndarray)
                    and val.shape[0] != inferred_batch_size):
                raise ValueError(
                    'Each entry in agent_infos must have a batch dimension of '
                    'length {}, but got key {} with batch size {} instead.'.
                    format(inferred_batch_size, key, val.shape[0]))

        # step_types
        if step_types.shape != (inferred_batch_size, ):
            raise ValueError(
                'step_types tensor must have shape {}, but got shape {} '
                'instead.'.format(inferred_batch_size, step_types.shape))

        if step_types.dtype != StepType:
            raise ValueError(
                'step_types tensor must be dtype `StepType`, but got tensor '
                'of dtype {} instead.'.format(step_types.dtype))

        return super().__new__(EpisodeBatch, env_spec, observations,
                               last_observations, actions, rewards, env_infos,
                               agent_infos, step_types, lengths)

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
        return cls(
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

    def split(self):
        """Split an EpisodeBatch into a list of EpisodeBatches.

        The opposite of concatenate.

        Returns:
            list[EpisodeBatch]: A list of EpisodeBatches, with one
                episode per batch.

        """
        episodes = []
        start = 0
        for i, length in enumerate(self.lengths):
            stop = start + length
            eps = EpisodeBatch(env_spec=self.env_spec,
                               observations=self.observations[start:stop],
                               last_observations=np.asarray(
                                   [self.last_observations[i]]),
                               actions=self.actions[start:stop],
                               rewards=self.rewards[start:stop],
                               env_infos=tensor_utils.slice_nested_dict(
                                   self.env_infos, start, stop),
                               agent_infos=tensor_utils.slice_nested_dict(
                                   self.agent_infos, start, stop),
                               step_types=self.step_types[start:stop],
                               lengths=np.asarray([length]))
            episodes.append(eps)
            start = stop
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
                * actions (np.ndarray): Non-flattened array of actions. Should
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
        start = 0
        episodes = []
        for i, length in enumerate(self.lengths):
            stop = start + length
            episodes.append({
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
            start = stop
        return episodes

    @classmethod
    def from_list(cls, env_spec, paths):
        """Create a EpisodeBatch from a list of episodes.

        Args:
            env_spec (EnvSpec): Specification for the environment from which
                this data was sampled.
            paths (list[dict[str, np.ndarray or dict[str, np.ndarray]]]): Keys:
                * observations (np.ndarray): Non-flattened array of
                    observations. Typically has shape (T, S^*) (the unflattened
                    state space of the current environment). observations[i]
                    was used by the agent to choose actions[i]. observations
                    may instead have shape (T + 1, S^*).
                * next_observations (np.ndarray): Non-flattened array of
                    observations. Has shape (T, S^*). next_observations[i] was
                    observed by the agent after taking actions[i]. Optional.
                    Note that to ensure all information from the environment
                    was preserved, observations[i] should have shape (T + 1,
                    S^*), or this key should be set. However, this method is
                    lenient and will "duplicate" the last observation if the
                    original last observation has been lost.
                * actions (np.ndarray): Non-flattened array of actions. Should
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

        stacked_paths = tensor_utils.concat_tensor_dict_list(paths)

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
        """Get the observations seen after actions are performed.

        Usually, in an :class:`~EpisodeBatch`, next_observations don't need to
        be stored explicitly, since the next observation is already stored in
        the batch.

        Returns:
            np.ndarray: The "next_observations".

        """
        return np.concatenate(
            tuple([
                np.concatenate((eps.observations[1:], eps.last_observations))
                for eps in self.split()
            ]))


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


class TimeStep(
        collections.namedtuple('TimeStep', [
            'env_spec', 'observation', 'action', 'reward', 'next_observation',
            'env_info', 'agent_info', 'step_type'
        ])):
    # pylint: disable=missing-return-doc, missing-return-type-doc, missing-param-doc, missing-type-doc  # noqa: E501
    r"""A tuple representing a single TimeStep.

    A :class:`~TimeStep` represents a single sample when an agent interacts
        with an environment. It describes as SARS (State–action–reward–state)
        tuple that characterizes the evolution of a MDP.

    Attributes:
        env_spec (EnvSpec): Specification for the environment from which this
            data was sampled.
        observation (numpy.ndarray): A numpy array of shape :math:`(O^*)`
            containing the observation for the this time step in the
            environment. These must conform to
            :obj:`EnvStep.observation_space`.
            The observation before applying the action.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        action (numpy.ndarray): A numpy array of shape :math:`(A^*)`
            containing the action for the this time step. These must conform
            to :obj:`EnvStep.action_space`.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        reward (float): A float representing the reward for taking the action
            given the observation, at the this time step.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        next_observation (numpy.ndarray): A numpy array of shape :math:`(O^*)`
            containing the observation for the this time step in the
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
    def from_env_step(cls, env_step, last_observation, agent_info):
        """Create a TimeStep from a EnvStep.

        Args:
            env_step (EnvStep): the env step returned by the environment.
            last_observation (numpy.ndarray): A numpy array of shape
                :math:`(O^*)` containing the observation for the this time
                step in the environment. These must conform to
                :obj:`EnvStep.observation_space`.
                The observation before applying the action.
            agent_info (dict):  A dict of arbitrary agent state information.

        Returns:
            TimeStep: The TimeStep with all information of EnvStep plus the
            agent info.
        """
        return cls(env_spec=env_step.env_spec,
                   observation=last_observation,
                   action=env_step.action,
                   reward=env_step.reward,
                   next_observation=env_step.observation,
                   env_info=env_step.env_info,
                   agent_info=agent_info,
                   step_type=env_step.step_type)


class InOutSpec:
    """Describes the input and output spaces of a primitive or module.

    Args:
        input_space (akro.Space): Input space of a module.
        output_space (akro.Space): Output space of a module.

    """

    def __init__(self, input_space, output_space):
        self._input_space = input_space
        self._output_space = output_space

    @property
    def input_space(self):
        """Get input space of the module.

        Returns:
            akro.Space: Input space of the module.

        """
        return self._input_space

    @property
    def output_space(self):
        """Get output space of the module.

        Returns:
            akro.Space: Output space of the module.

        """
        return self._output_space


class TimeStepBatch(
        collections.namedtuple('TimeStepBatch', [
            'env_spec', 'observations', 'actions', 'rewards',
            'next_observations', 'env_infos', 'agent_infos', 'step_types'
        ])):
    # pylint: disable=missing-param-doc, missing-type-doc
    """A tuple representing a batch of TimeSteps.

    Data type for off-policy algorithms, imitation learning and batch-RL.

    Attributes:
        env_spec (EnvSpec): Specification for the environment from
            which this data was sampled.
        observations (numpy.ndarray): Non-flattened array of observations.
            Typically has shape (batch_size, S^*) (the unflattened state space
            of the current environment).
        actions (numpy.ndarray): Non-flattened array of actions. Should
            have shape (batch_size, S^*) (the unflattened action space of the
            current environment).
        rewards (numpy.ndarray): Array of rewards of shape (batch_size,) (1D
            array of length batch_size).
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
    __slots__ = ()

    def __new__(cls, env_spec, observations, actions, rewards,
                next_observations, env_infos, agent_infos,
                step_types):  # noqa: D102
        # pylint: disable=missing-return-doc, missing-return-type-doc,
        # pylint: disable=too-many-branches

        inferred_batch_size = len(rewards)
        if inferred_batch_size < 1:
            raise ValueError(
                'Expected batch dimension of rewards to be greater than 1, '
                'but got length {} instead.'.format(inferred_batch_size))

        first_observation = observations[0]
        first_action = actions[0]

        # observation
        if not env_spec.observation_space.contains(first_observation):
            if isinstance(env_spec.observation_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.observation_space.flat_dim != np.prod(
                        first_observation.shape):
                    raise ValueError('observations should have the same '
                                     'dimensionality as the observation_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.observation_space.flat_dim,
                                         first_observation.shape))
            else:
                raise ValueError(
                    'observations must conform to observation_space {}, '
                    'but got data with shape {} instead.'.format(
                        env_spec.observation_space, first_observation.shape))
        if observations.shape[0] != inferred_batch_size:
            raise ValueError(
                'Expected batch dimension of observations to be length {}, '
                'but got length {} instead.'.format(inferred_batch_size,
                                                    observations.shape[0]))

        # next_observation
        if not env_spec.observation_space.contains(next_observations[0]):
            if isinstance(env_spec.observation_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.observation_space.flat_dim != np.prod(
                        next_observations[0].shape):
                    raise ValueError('next_observations should have the same '
                                     'dimensionality as the observation_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.observation_space.flat_dim,
                                         next_observations[0].shape))
            else:
                raise ValueError(
                    'next_observations must conform to observation_space {}, '
                    'but got data with shape {} instead.'.format(
                        env_spec.observation_space,
                        next_observations[0].shape[0]))
        if next_observations.shape[0] != inferred_batch_size:
            raise ValueError(
                'Expected batch dimension of next_observations to be length {'
                '}, but got length {} instead.'.format(
                    inferred_batch_size, next_observations[0].shape[0]))

        # action
        if not env_spec.action_space.contains(first_action):
            if isinstance(env_spec.action_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.action_space.flat_dim != np.prod(
                        first_action.shape):
                    raise ValueError('actions should have the same '
                                     'dimensionality as the action_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.action_space.flat_dim,
                                         first_action.shape))
            else:
                raise ValueError('actions must conform to action_space {}, '
                                 'but got data with shape {} instead.'.format(
                                     env_spec.action_space,
                                     first_action.shape))
        if actions.shape[0] != inferred_batch_size:
            raise ValueError(
                'Expected batch dimension of actions to be length {}, but got '
                'length {} instead.'.format(inferred_batch_size,
                                            actions.shape[0]))

        # step_types
        if step_types.shape[0] != inferred_batch_size:
            raise ValueError(
                'Expected batch dimension of step_types to be length {}, '
                'but got '
                'length {} instead.'.format(inferred_batch_size,
                                            rewards.shape[0]))

        for step_type in step_types:
            if not isinstance(step_type, StepType):
                raise ValueError(
                    'Each entry in step_types must be a StepType, but got'
                    ' value type {} instead.'.format(type(step_type)))

        # env_infos
        for key, val in env_infos.items():
            if not isinstance(val, (dict, np.ndarray)):
                raise ValueError(
                    'Each entry in env_infos must be a numpy array or '
                    'dictionary, but got key {} with value type {} '
                    'instead.'.format(key, type(val)))

            if (isinstance(val, np.ndarray)
                    and val.shape[0] != inferred_batch_size):
                raise ValueError(
                    'Each entry in env_infos must have a batch dimension '
                    'of '
                    'length {}, but got key {} with batch size {} instead.'.
                    format(inferred_batch_size, key, val.shape[0]))

        # agent_infos
        for key, val in agent_infos.items():
            if not isinstance(val, (dict, np.ndarray)):
                raise ValueError(
                    'Each entry in agent_infos must be a numpy array or '
                    'dictionary, but got key {} with value type {} instead.'
                    'instead'.format(key, type(val)))

            if (isinstance(val, np.ndarray)
                    and val.shape[0] != inferred_batch_size):
                raise ValueError(
                    'Each entry in agent_infos must have a batch '
                    'dimension of '
                    'length {}, but got key {} with batch size {} instead.'.
                    format(inferred_batch_size, key, val.shape[0]))

        return super().__new__(TimeStepBatch, env_spec, observations, actions,
                               rewards, next_observations, env_infos,
                               agent_infos, step_types)

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
            observations=np.concatenate(
                [batch.observations for batch in batches]),
            actions=np.concatenate([batch.actions for batch in batches]),
            rewards=np.concatenate([batch.rewards for batch in batches]),
            next_observations=np.concatenate(
                [batch.next_observations for batch in batches]),
            env_infos=env_infos,
            agent_infos=agent_infos,
            step_types=np.concatenate([batch.step_types for batch in batches]))

    def split(self):
        """Split a :class:`~TimeStepBatch` into a list of :class:`~TimeStepBatch`s.

        The opposite of concatenate.

        Returns:
            list[TimeStepBatch]: A list of :class:`TimeStepBatch`s, with one
                :class:`~TimeStep` per :class:`~TimeStepBatch`.

        """
        time_steps = []

        for i in range(len(self.rewards)):
            time_step = TimeStepBatch(
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

    def to_time_step_list(self):
        """Convert the batch into a list of dictionaries.

        Breaks the :class:`~TimeStepBatch` into a list of single time step
        sample dictionaries. len(rewards) (or the number of discrete time step)
        dictionaries are returned

        Returns:
            list[dict[str, np.ndarray or dict[str, np.ndarray]]]: Keys:
                observations (numpy.ndarray): Non-flattened array of
                    observations.
                    Typically has shape (batch_size, S^*) (the unflattened
                    state space
                    of the current environment).
                actions (numpy.ndarray): Non-flattened array of actions. Should
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
                np.asarray([self.step_types[i]]),
            })
        return samples

    @classmethod
    def from_time_step_list(cls, env_spec, ts_samples):
        """Create a :class:`~TimeStepBatch` from a list of time step dictionaries.

        Args:
            env_spec (EnvSpec): Specification for the environment from which
                this data was sampled.
            ts_samples (list[dict[str, np.ndarray or dict[str, np.ndarray]]]):
                keys:
                * observations (numpy.ndarray): Non-flattened array of
                    observations.
                    Typically has shape (batch_size, S^*) (the unflattened
                    state space of the current environment).
                * actions (numpy.ndarray): Non-flattened array of actions.
                    Should have shape (batch_size, S^*) (the unflattened action
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
            TimeStepBatch(env_spec=env_spec,
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

    @classmethod
    def from_episode_batch(cls, batch):
        """Construct a :class:`~TimeStepBatch` from an :class:`~EpisodeBatch`.

        Args:
            batch (EpisodeBatch): Episode batch to convert.

        Returns:
            TimeStepBatch: The converted batch.

        """
        next_observations = np.concatenate(
            tuple([
                np.concatenate((eps.observations[1:], eps.last_observations))
                for eps in batch.split()
            ]))
        return cls(env_spec=batch.env_spec,
                   observations=batch.observations,
                   actions=batch.actions,
                   rewards=batch.rewards,
                   next_observations=next_observations,
                   env_infos=batch.env_infos,
                   agent_infos=batch.agent_infos,
                   step_types=batch.step_types)
