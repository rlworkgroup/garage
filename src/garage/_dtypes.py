"""Data types for agent-based learning."""
import collections

import numpy as np


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
            raise ValueError(
                'observations must conform to observation_space {}, but got '
                'data with shape {} instead.'.format(
                    env_spec.observation_space, first_observation))

        if observations.shape[0] != inferred_batch_size:
            raise ValueError(
                'Expected batch dimension of observations to be length {}, '
                'but got length {} instead.'.format(inferred_batch_size,
                                                    observations.shape[0]))

        # last observations
        if not env_spec.observation_space.contains(last_observations[0]):
            raise ValueError(
                'last_observations must conform to observation_space {}, but '
                'got ata with shape {} instead.'.format(
                    env_spec.observation_space, first_observation))

        if last_observations.shape[0] != len(lengths):
            raise ValueError(
                'Expected batch dimension of last_observations to be length '
                '{}, but got length {} instead.'.format(
                    len(lengths), last_observations.shape[0]))

        # actions
        if not env_spec.action_space.contains(first_action):
            raise ValueError(
                'actions must conform to action_space {}, but got data with '
                'shape {} instead.'.format(env_spec.action_space,
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
            if not isinstance(val, np.ndarray):
                raise ValueError(
                    'Each entry in env_infos must be a numpy array, but got '
                    'key {} with value type {} instead.'
                    'instead'.format(key, type(val)))

            if val.shape[0] != inferred_batch_size:
                raise ValueError(
                    'Each entry in env_infos must have a batch dimension of '
                    'length {}, but got key {} with batch size {} instead.'.
                    format(inferred_batch_size, key, val.shape[0]))

        # agent_infos
        for key, val in agent_infos.items():
            if not isinstance(val, np.ndarray):
                raise ValueError(
                    'Each entry in agent_infos must be a numpy array, but got '
                    'key {} with value type {} instead.'
                    'instead'.format(key, type(val)))

            if val.shape[0] != inferred_batch_size:
                raise ValueError(
                    'Each entry in agent_infos must have a batch dimension of '
                    'length {}, but got key {} with batch size {} instead.'.
                    format(inferred_batch_size, key, val.shape[0]))

        return super().__new__(TrajectoryBatch, env_spec, observations,
                               last_observations, actions, rewards, terminals,
                               env_infos, agent_infos, lengths)


class TimeStep(
        collections.namedtuple('TimeStep', [
            'env_spec',
            'observation',
            'action',
            'reward',
            'next_observation',
            'terminal',
            'env_info',
            'agent_info',
        ])):
    # pylint: disable=missing-return-doc, missing-return-type-doc, missing-param-doc, missing-type-doc  # noqa: E501
    r"""A tuple representing a single TimeStep.

    A :class:`TimeStep` represents a single sample when an agent interacts with
        an environment.

    Attributes:
        env_spec (garage.envs.EnvSpec): Specification for the environment from
            which this data was sampled.
        observation (numpy.ndarray): A numpy array of shape :math:`(O^*)`
            containing the observation for the this time step in the
            environment. These must conform to
            :obj:`env_spec.observation_space`.
        action (numpy.ndarray): A numpy array of shape :math:`(A^*)`
            containing the action for the this time step. These must conform
            to :obj:`env_spec.action_space`.
        reward (float): A float representing the reward for taking the action
            given the observation, at the this time step.
        terminals (bool): The termination signal for the this time step.
        env_info (dict): A dict arbitrary environment state information.
        agent_info (numpy.ndarray): A dict of arbitrary agent
            state information. For example, this may contain the hidden states
            from an RNN policy.


    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """

    def __new__(cls, env_spec, observation, action, reward, next_observation,
                terminal, env_info, agent_info):  # noqa: D102

        # observation
        if not env_spec.observation_space.contains(observation):
            raise ValueError(
                'observation must conform to observation_space {}, but got '
                'data with shape {} instead.'.format(
                    env_spec.observation_space, observation))

        if not env_spec.observation_space.contains(next_observation):
            raise ValueError(
                'next_observation must conform to observation_space {},'
                ' but got data with shape {} instead.'.format(
                    env_spec.observation_space, next_observation))

        # action
        if not env_spec.action_space.contains(action):
            raise ValueError(
                'action must conform to action_space {}, but got data with '
                'shape {} instead.'.format(env_spec.action_space, action))

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

        return super().__new__(TimeStep, env_spec, observation, action, reward,
                               next_observation, terminal, env_info,
                               agent_info)
