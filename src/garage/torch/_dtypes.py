"""Data structures used in garage.torch."""
from dataclasses import dataclass
import enum

import torch
from torch import nn


class ShuffledOptimizationNotSupported(ValueError):
    """Raised by recurrent policies if they're passed a shuffled batch."""


class ObservationOrder(enum.IntEnum):
    """Defines the order of observations in an ObservationBatch.

    See :class:`ObservationBatch` for detailed documentation.

    """
    # Tensor contains a batch of "most recent" observations.
    # This ordering is typcially used when performing rollouts, and it is
    # expected that stateful policies maintain there own state when using this
    # ordering.
    LAST = 0
    # Tensor contains observations with timesteps from potentially different
    # episodes in a shuffled order. Recurrent policies should raise
    # ShuffledOptimizationNotSupported if they encounter this ordering.
    SHUFFLED = 1
    # Tensor contains all observations for a batch of episodes, in order.
    EPISODES = 2


@dataclass(init=False, eq=False)
class ObservationBatch(torch.Tensor):
    r"""The (differentiable) input to all pytorch policies.

    Args:
        observations (torch.Tensor): A torch tensor containing flattened
            observations in a batch. Stateless policies should always operate
            on this input. This input is passed to the super-constructor.
            Shape depends on the order:
             * If `order == ROLLOUT`, has shape :math:`(V, O)` (where V is the
                vectorization level).
             * If `order == SHUFFLED`, has shape :math:`(B, O)` (where B is the
                mini-batch size).
             * If order == EPISODES, has shape :math:`(N \bullet [T], O)`
                (where N is the number of episodes, and T is the episode
                lengths).
        order (ObservationOrder): The order of observations in this batch. If
            this is set to EPISODES, lengths must not be None.
        lengths (torch.Tensor or None): Integer tensor containing the lengths
            of each episode. Only has a value if `order == EPISODES`.
    """

    order: ObservationOrder
    lengths: torch.Tensor = None

    def __new__(cls, observations, order, lengths=None):
        """Check that lengths is consistent with the rest of the fields.

        Raises:
            ValueError: If lengths is not consistent with another field.

        Returns:
            ObservationBatch: A new observation batch.

        """
        self = super().__new__(cls, observations)
        self.order = order
        self.lengths = lengths
        if self.order == ObservationOrder.EPISODES:
            if self.lengths is None:
                raise ValueError(
                    'lengths is None, but must be a torch.Tensor when '
                    'order == ObservationOrder.EPISODES')
            assert self.lengths is not None
            if self.lengths.dtype not in (torch.uint8, torch.int8, torch.int16,
                                          torch.int32, torch.int64):
                raise ValueError(
                    f'lengths has dtype {self.lengths.dtype}, but must have '
                    f'an integer dtype')
            total_size = sum(self.lengths)
            if self.shape[0] != total_size:
                raise ValueError(
                    f'observations has batch size '
                    f'{self.observations.shape[0]}, but must have batch '
                    f'size {total_size} to match lengths')
            assert self.shape[0] == total_size
        elif self.lengths is not None:
            raise ValueError(
                f'lengths has value {self.lengths}, but must be None '
                f'when order == {self.order}')
        return self

    def __repr__(self):
        return f'{type(self).__name__}({super().__repr__()}, order={self.order!r}, lengths={self.lengths!r})'

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # print(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
        if kwargs is None:
            kwargs = {}
        result = super().__torch_function__(func, types, args, kwargs)
        # Fixup ObservationBatch instances returned from methods.
        # In the future this might preserve order for some methods
        if isinstance(result, ObservationBatch):
            if not hasattr(result, 'order'):
                result.order = ObservationOrder.SHUFFLED
            if not hasattr(result, 'lengths'):
                result.lengths = None
        return result


def observation_batch_to_packed_sequence(observations):
    """Turn ObservationBatch into a torch.nn.utils.rnn.PackedSequence.

    This function is not a method on ObservationBatch so that it can be called
    on a observation Tensor that is not an ObservationBatch. This simplifies
    the implementation of recurrent policies.

    Args:
        observations (torch.Tensor or ObservationBatch): Observations to
            convert to PackedSequence.

    Raises:
        ShuffledOptimizationNotSupported: If called with an input that is not
            an ObservationBatch or when `order != EPISODES`

    Returns:
        torch.nn.utils.rnn.PackedSequence: The sequence of flattened
            observations.

    """
    if not isinstance(observations, ObservationBatch):
        raise ShuffledOptimizationNotSupported(
            f'observations should be an ObservationBatch, but was of '
            f'type {type(observations)!r} instead.')
    if observations.order != ObservationOrder.EPISODES:
        raise ShuffledOptimizationNotSupported(
            f'order has value {observations.order} but must have order '
            f'{ObservationOrder.EPISODES} to use to_packed_sequence')
    sequence = []
    start = 0
    for length in observations.lengths:
        stop = start + length
        sequence.append(observations.observations[start:stop])
        start = stop
    pack_sequence = nn.utils.rnn.pack_sequence
    return pack_sequence(sequence, enforce_sorted=False)


def is_policy_recurrent(policy, env_spec):
    """Check if a torch policy is recurrent.

    Args:
        policy (garage.torch.Policy): Policy that might be recurrent.

    Returns:
        bool: If policy is recurrent.

    """
    try:
        policy.forward(
            as_tensor([
                env_spec.observation_space.sample(),
                env_spec.observation_space.sample()
            ]))
    except ShuffledOptimizationNotSupported:
        return True
    else:
        return False
