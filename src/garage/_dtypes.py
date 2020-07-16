"""Data types for agent-based learning."""
import collections
import enum

import akro
import numpy as np

from garage.misc import tensor_utils

# pylint: disable=too-many-lines


class TrajectoryBatch(
        collections.namedtuple('TrajectoryBatch', [
            'env_spec',
            'observations',
            'last_observations',
            'actions',
            'rewards',
            'terminals',
            'env_infos',
            'agent_infos',
            'lengths',
        ])):
    # pylint: disable=missing-return-doc, missing-return-type-doc, missing-param-doc, missing-type-doc  # noqa: E501
    r"""A tuple representing a batch of whole trajectories.

    Data type for on-policy algorithms.

    A :class:`TrajectoryBatch` represents a batch of whole trajectories
    produced when one or more agents interacts with one or more environments.

    +-----------------------+-------------------------------------------------+
    | Symbol                | Description                                     |
    +=======================+=================================================+
    | :math:`N`             | Trajectory index dimension                      |
    +-----------------------+-------------------------------------------------+
    | :math:`[T]`           | Variable-length time dimension of each          |
    |                       | trajectory                                      |
    +-----------------------+-------------------------------------------------+
    | :math:`S^*`           | Single-step shape of a time-series tensor       |
    +-----------------------+-------------------------------------------------+
    | :math:`N \bullet [T]` | A dimension computed by flattening a            |
    |                       | variable-length time dimension :math:`[T]` into |
    |                       | a single batch dimension with length            |
    |                       | :math:`sum_{i \in N} [T]_i`                     |
    +-----------------------+-------------------------------------------------+

    Attributes:
        env_spec (garage.envs.EnvSpec): Specification for the environment from
            which this data was sampled.
        observations (numpy.ndarray): A numpy array of shape
            :math:`(N \bullet [T], O^*)` containing the (possibly
            multi-dimensional) observations for all time steps in this batch.
            These must conform to :obj:`env_spec.observation_space`.
        last_observations (numpy.ndarray): A numpy array of shape
            :math:`(N, O^*)` containing the last observation of each
            trajectory.  This is necessary since there are one more
            observations than actions every trajectory.
        actions (numpy.ndarray): A  numpy array of shape
            :math:`(N \bullet [T], A^*)` containing the (possibly
            multi-dimensional) actions for all time steps in this batch. These
            must conform to :obj:`env_spec.action_space`.
        rewards (numpy.ndarray): A numpy array of shape
            :math:`(N \bullet [T])` containing the rewards for all time steps
            in this batch.
        terminals (numpy.ndarray): A boolean numpy array of shape
            :math:`(N \bullet [T])` containing the termination signals for all
            time steps in this batch.
        env_infos (dict): A dict of numpy arrays arbitrary environment state
            information. Each value of this dict should be a numpy array of
            shape :math:`(N \bullet [T])` or :math:`(N \bullet [T], S^*)`.
        agent_infos (numpy.ndarray): A dict of numpy arrays arbitrary agent
            state information. Each value of this dict should be a numpy array
            of shape :math:`(N \bullet [T])` or :math:`(N \bullet [T], S^*)`.
            For example, this may contain the hidden states from an RNN policy.
        lengths (numpy.ndarray): An integer numpy array of shape :math:`(N,)`
            containing the length of each trajectory in this batch. This may be
            used to reconstruct the individual trajectories.

    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """
    __slots__ = ()

    def __new__(cls, env_spec, observations, last_observations, actions,
                rewards, terminals, env_infos, agent_infos,
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

        # terminals
        if terminals.shape != (inferred_batch_size, ):
            raise ValueError(
                'terminals tensor must have shape {}, but got shape {} '
                'instead.'.format(inferred_batch_size, terminals.shape))

        if terminals.dtype != np.bool:
            raise ValueError(
                'terminals tensor must be dtype np.bool, but got tensor '
                'of dtype {} instead.'.format(terminals.dtype))

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

        return super().__new__(TrajectoryBatch, env_spec, observations,
                               last_observations, actions, rewards, terminals,
                               env_infos, agent_infos, lengths)

    @classmethod
    def concatenate(cls, *batches):
        """Create a TrajectoryBatch by concatenating TrajectoryBatches.

        Args:
            batches (list[TrajectoryBatch]): Batches to concatenate.

        Returns:
            TrajectoryBatch: The concatenation of the batches.

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
            batches[0].env_spec,
            np.concatenate([batch.observations for batch in batches]),
            np.concatenate([batch.last_observations for batch in batches]),
            np.concatenate([batch.actions for batch in batches]),
            np.concatenate([batch.rewards for batch in batches]),
            np.concatenate([batch.terminals for batch in batches]), env_infos,
            agent_infos, np.concatenate([batch.lengths for batch in batches]))

    def split(self):
        """Split a TrajectoryBatch into a list of TrajectoryBatches.

        The opposite of concatenate.

        Returns:
            list[TrajectoryBatch]: A list of TrajectoryBatches, with one
                trajectory per batch.

        """
        trajectories = []
        start = 0
        for i, length in enumerate(self.lengths):
            stop = start + length
            traj = TrajectoryBatch(env_spec=self.env_spec,
                                   observations=self.observations[start:stop],
                                   last_observations=np.asarray(
                                       [self.last_observations[i]]),
                                   actions=self.actions[start:stop],
                                   rewards=self.rewards[start:stop],
                                   terminals=self.terminals[start:stop],
                                   env_infos=tensor_utils.slice_nested_dict(
                                       self.env_infos, start, stop),
                                   agent_infos=tensor_utils.slice_nested_dict(
                                       self.agent_infos, start, stop),
                                   lengths=np.asarray([length]))
            trajectories.append(traj)
            start = stop
        return trajectories

    def to_trajectory_list(self):
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
                * dones (np.ndarray): Array of dones of shape (T,) (1D array
                    of length timesteps).
                * agent_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `agent_info` arrays.
                * env_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `env_info` arrays.

        """
        start = 0
        trajectories = []
        for i, length in enumerate(self.lengths):
            stop = start + length
            trajectories.append({
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
                'dones':
                self.terminals[start:stop]
            })
            start = stop
        return trajectories

    @classmethod
    def from_trajectory_list(cls, env_spec, paths):
        """Create a TrajectoryBatch from a list of trajectories.

        Args:
            env_spec (garage.envs.EnvSpec): Specification for the environment
                from which this data was sampled.
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
                * dones (np.ndarray): Array of rewards of shape (T,) (1D array
                    of length timesteps).
                * agent_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `agent_info` arrays.
                * env_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `env_info` arrays.

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
        return cls(env_spec=env_spec,
                   observations=observations,
                   last_observations=last_observations,
                   actions=stacked_paths['actions'],
                   rewards=stacked_paths['rewards'],
                   terminals=stacked_paths['dones'],
                   env_infos=stacked_paths['env_infos'],
                   agent_infos=stacked_paths['agent_infos'],
                   lengths=lengths)

    @property
    def next_observations(self):
        """Get the observations seen after actions are performed.

        Usually, in a TrajectoryBatch, next_observations don't need to be
        stored explicitly, since the next observation is already stored in the
        batch.

        Returns:
            np.ndarray: The "next_observations".

        """
        return np.concatenate(
            tuple([
                np.concatenate((traj.observations[1:], traj.last_observations))
                for traj in self.split()
            ]))


class StepType(enum.IntEnum):
    """Defines the status of a `TimeStep` within a sequence."""
    # Denotes the first `TimeStep` in a sequence.
    FIRST = 0
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = 1
    # Denotes the last `TimeStep` in a sequence.
    LAST = 2
    # Denotes the `TimeStep` truncated by time limit
    TIMEOUT = 3


class TimeStep(
        collections.namedtuple('TimeStep', [
            'env_spec', 'observation', 'action', 'reward', 'next_observation',
            'terminal', 'env_info', 'agent_info', 'step_type'
        ])):
    # pylint: disable=missing-return-doc, missing-return-type-doc, missing-param-doc, missing-type-doc  # noqa: E501
    r"""A tuple representing a single TimeStep.

    A :class:`TimeStep` represents a single sample when an agent interacts with
        an environment. It describes as SARS (State–action–reward–state) tuple
        that characterizes the evolution of a MDP.

    Attributes:
        env_spec (garage.envs.EnvSpec): Specification for the environment from
            which this data was sampled.
        observation (numpy.ndarray): A numpy array of shape :math:`(O^*)`
            containing the observation for the this time step in the
            environment. These must conform to
            :obj:`env_spec.observation_space`.
            The observation before applying the action.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        action (numpy.ndarray): A numpy array of shape :math:`(A^*)`
            containing the action for the this time step. These must conform
            to :obj:`env_spec.action_space`.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        reward (float): A float representing the reward for taking the action
            given the observation, at the this time step.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        next_observation (numpy.ndarray): A numpy array of shape :math:`(O^*)`
            containing the observation for the this time step in the
            environment. These must conform to
            :obj:`env_spec.observation_space`.
            The observation after applying the action.
        terminal (bool): The termination signal for the this time step.
        env_info (dict): A dict arbitrary environment state information.
        agent_info (numpy.ndarray): A dict of arbitrary agent
            state information. For example, this may contain the hidden states
            from an RNN policy.
        step_type (garage.StepType): a `StepType` enum value. Can either be
            StepType.FIRST, StepType.MID, StepType.LAST, StepType.TIMEOUT.


    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """

    def __new__(cls, env_spec, observation, action, reward, next_observation,
                terminal, env_info, agent_info, step_type):  # noqa: D102
        # pylint: disable=too-many-branches
        # observation
        if not env_spec.observation_space.contains(observation):
            if isinstance(env_spec.observation_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.observation_space.flat_dim != np.prod(
                        observation.shape):
                    raise ValueError('observation should have the same '
                                     'dimensionality as the observation_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.observation_space.flat_dim,
                                         observation.shape))
            else:
                raise ValueError(
                    'observation must conform to observation_space {}, '
                    'but got data with shape {} instead.'.format(
                        env_spec.observation_space, observation))

        if not env_spec.observation_space.contains(next_observation):
            if isinstance(env_spec.observation_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.observation_space.flat_dim != np.prod(
                        next_observation.shape):
                    raise ValueError('next_observation should have the same '
                                     'dimensionality as the observation_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.observation_space.flat_dim,
                                         next_observation.shape))
            else:
                raise ValueError(
                    'next_observation must conform to observation_space {}, '
                    'but got data with shape {} instead.'.format(
                        env_spec.observation_space, next_observation))

        # action
        if not env_spec.action_space.contains(action):
            if isinstance(env_spec.action_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.action_space.flat_dim != np.prod(action.shape):
                    raise ValueError('action should have the same '
                                     'dimensionality as the action_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.action_space.flat_dim,
                                         action.shape))
            else:
                raise ValueError('action must conform to action_space {}, '
                                 'but got data with shape {} instead.'.format(
                                     env_spec.action_space, action))

        if not isinstance(agent_info, dict):
            raise ValueError('agent_info must be type {}, but got type {} '
                             'instead.'.format(dict, type(agent_info)))

        if not isinstance(env_info, dict):
            raise ValueError('env_info must be type {}, but got type {} '
                             'instead.'.format(dict, type(env_info)))

        # rewards
        if not isinstance(reward, float):
            raise ValueError('reward must be type {}, but got type {} '
                             'instead.'.format(float, type(reward)))

        if not isinstance(terminal, bool):
            raise ValueError(
                'terminal must be dtype bool, but got dtype {} instead.'.
                format(type(terminal)))

        if not isinstance(step_type, StepType):
            raise ValueError(
                'step_type must be dtype garage.StepType, but got dtype {} '
                'instead.'.format(type(step_type)))

        return super().__new__(TimeStep, env_spec, observation, action, reward,
                               next_observation, terminal, env_info,
                               agent_info, step_type)

    def first(self):
        """Checks if the `TimeStep` is the first one in a sequence.

        Returns:
            bool: True if it is a FIRST `TimeStep`.
        """
        return self.step_type is StepType.FIRST

    def mid(self):
        """Checks if the `TimeStep` is the mid one in a sequence.

        Returns:
          bool: True if it is a MID `TimeStep`.
        """
        return self.step_type is StepType.MID

    def last(self):
        """Checks if the `TimeStep` is the last one in a successful sequence.

        Returns:
          bool: True if it is a LAST `TimeStep`.
        """
        return self.step_type is StepType.LAST

    def timeout(self):
        """Checks if the `TimeStep` is terminated by time out.

        Returns:
          bool: True if it is a TIMEOUT `TimeStep`.
        """
        return self.step_type is StepType.TIMEOUT


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
            'env_spec',
            'observations',
            'actions',
            'rewards',
            'next_observations',
            'terminals',
            'env_infos',
            'agent_infos',
        ])):
    # pylint: disable=missing-param-doc, missing-type-doc
    """A tuple representing a batch of TimeSteps.

    Data type for off-policy algorithms, imitation learning and batch-RL.

    Attributes:
        env_spec (garage.envs.EnvSpec): Specification for the environment from
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
        terminals (numpy.ndarray): A boolean numpy array of shape
            shape (batch_size,) containing the termination signals for all
            transitions in this batch.
        env_infos (dict): A dict arbitrary environment state
            information.
        agent_infos (dict): A dict of arbitrary agent state information. For
            example, this may contain the hidden states from an RNN policy.

    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """
    __slots__ = ()

    def __new__(cls, env_spec, observations, actions, rewards,
                next_observations, terminals, env_infos,
                agent_infos):  # noqa: D102
        # pylint: disable=missing-return-doc, missing-return-type-doc,
        # pylint: disable=too-many-branches

        inferred_batch_size = len(terminals)
        if inferred_batch_size < 1:
            raise ValueError(
                'Expected batch dimension of terminals to be greater than 1, '
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

        # rewards
        if rewards.shape[0] != inferred_batch_size:
            raise ValueError(
                'Expected batch dimension of rewards to be length {}, but got '
                'length {} instead.'.format(inferred_batch_size,
                                            rewards.shape[0]))

        # terminals
        if terminals.dtype != np.bool:
            raise ValueError(
                'terminals tensor must be dtype np.bool, but got tensor '
                'of dtype {} instead.'.format(terminals.dtype))

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
                               rewards, next_observations, terminals,
                               env_infos, agent_infos)

    @classmethod
    def concatenate(cls, *batches):
        """Create a TimeStepBatch by concatenating TimeStepBatches.

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
            batches[0].env_spec,
            np.concatenate([batch.observations for batch in batches]),
            np.concatenate([batch.actions for batch in batches]),
            np.concatenate([batch.rewards for batch in batches]),
            np.concatenate([batch.next_observations for batch in batches]),
            np.concatenate([batch.terminals for batch in batches]), env_infos,
            agent_infos)

    def split(self):
        """Split a TimeStepBatch into a list of TimeStepBatches.

        The opposite of concatenate.

        Returns:
            list[TimeStepBatch]: A list of TimeStepBatches, with one
                TimeStep per TimeStepBatch.

        """
        time_steps = []

        for i in range(len(self.terminals)):
            time_step = TimeStepBatch(
                env_spec=self.env_spec,
                observations=np.asarray([self.observations[i]]),
                actions=np.asarray([self.actions[i]]),
                rewards=np.asarray([self.rewards[i]]),
                next_observations=np.asarray([self.next_observations[i]]),
                terminals=np.asarray([self.terminals[i]]),
                env_infos={
                    k: np.asarray([v[i]])
                    for (k, v) in self.env_infos.items()
                },
                agent_infos={
                    k: np.asarray([v[i]])
                    for (k, v) in self.agent_infos.items()
                },
            )
            time_steps.append(time_step)
        return time_steps

    def to_time_step_list(self):
        """Convert the batch into a list of dictionaries.

        This breaks the TimeStepBatch object into a list of single
        time step sample dictionaries. len(terminals) (or the number of
        discrete time step) dictionaries are returned


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
                terminals (numpy.ndarray): A boolean numpy array of shape
                    shape (batch_size,) containing the termination signals
                    for all
                    transitions in this batch.
                env_infos (dict): A dict arbitrary environment state
                    information.
                agent_infos (dict): A dict of arbitrary agent state
                    information. For example, this may contain the
                    hidden states from an RNN policy.

        """
        samples = []
        for i in range(len(self.terminals)):
            samples.append({
                'observations':
                np.asarray([self.observations[i]]),
                'actions':
                np.asarray([self.actions[i]]),
                'rewards':
                np.asarray([self.rewards[i]]),
                'next_observations':
                np.asarray([self.next_observations[i]]),
                'terminals':
                np.asarray([self.terminals[i]]),
                'env_infos':
                {k: np.asarray([v[i]])
                 for (k, v) in self.env_infos.items()},
                'agent_infos':
                {k: np.asarray([v[i]])
                 for (k, v) in self.agent_infos.items()},
            })
        return samples

    @classmethod
    def from_time_step_list(cls, env_spec, ts_samples):
        """Create a TimeStepBatch from a list of time step dictionaries.

        Args:
            env_spec (garage.envs.EnvSpec): Specification for the environment
                from which this data was sampled.
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
                * terminals (numpy.ndarray): A boolean numpy array of shape
                    shape (batch_size,) containing the termination signals
                    for all transitions in this batch.
                * env_infos (dict): A dict arbitrary environment state
                    information.
                * agent_infos (dict): A dict of arbitrary agent
                    state information. For example, this may contain the
                    hidden states from an RNN policy.

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
                          terminals=sample['terminals'],
                          env_infos=sample['env_infos'],
                          agent_infos=sample['agent_infos'])
            for sample in ts_samples
        ]

        return TimeStepBatch.concatenate(*ts_batches)

    @classmethod
    def from_trajectory_batch(cls, batch):
        """Construct a TimeStepBatch from a TrajectoryBatch.

        Args:
            batch (TrajectoryBatch): TrajectoryBatch to convert.

        Returns:
            TimeStepBatch: The converted batch.

        """
        next_observations = np.concatenate(
            tuple([
                np.concatenate((traj.observations[1:], traj.last_observations))
                for traj in batch.split()
            ]))
        return cls(env_spec=batch.env_spec,
                   observations=batch.observations,
                   actions=batch.actions,
                   rewards=batch.rewards,
                   next_observations=next_observations,
                   terminals=batch.terminals,
                   env_infos=batch.env_infos,
                   agent_infos=batch.agent_infos)
