"""This module implements a replay buffer memory.

Replay buffer is an important technique in reinforcement learning. It
stores transitions in a memory buffer of fixed size. When the buffer is
full, oldest memory will be discarded. At each step, a batch of memories
will be sampled from the buffer to update the agent's parameters. In a
word, replay buffer breaks temporal correlations and thus benefits RL
algorithms.

"""

import abc
from abc import abstractmethod

from typing import Sequence, Any, Union
import queue


class ReplayBuffer(abc.ABC):

    @abc.abstractmethod
    def insert(self, samples: Union[Any, Sequence[Any]]
               ) -> Union[Any, Sequence[str]]:
        """Insert samples into buffer.
            Returns a UUID that idenitifies each sample in the data store.
        """

    @abc.abstractmethod
    def sample(self, batch_size: int) -> Union[Sequence[Any], Any]:
        """Get samples from buffer."""


class PathBuffer(ReplayBuffer):

    def __init__(data_store, evic_policy_cls=FIFOEvictionPolicy):
        self.evic_policy = evic_policy_cls(data_store)
        self.data_store = data_store

    def insert(self, samples: TrajectoryBatch) -> Sequence[int]:
        return self.evic_policy.insert(samples.to_list())

    def sample(self, n: int) -> TrajectoryBatch:
        return TrajectoryBatch.from_list(self.data_store.sample(n))


class DataStore(abc.ABC):

    @abc.abstractmethod
    def insert(self, samples) -> Sequence[str]:
        """Insert samples into datastore. Returns keys/UUIDs for each sample."""

    @abc.abstractmethod
    def remove(self, key: str):
        """Remove samples"""

    @abc.abstractmethod
    def remove_batch(self, uuids: Sequence[str]):
        """Remove a batch of samples"""

    @property
    @abc.abstractmethod
    def is_full(self) -> bool:
        """True if at capacity, else false."""


class EvictionPolicy(abc.ABC):

    @abc.abstractmethod
    def insert(self, samples):
        'Evict samples if needed, then insert into data store.'


class FIFOEvictionPolicy(EvictionPolicy):

    def __init__(self, data_store):
        self.data_store = data_store
        self.uuids = queue.Queue()

    def insert(self, samples):
        if self.data_store.is_full():
            # evict some samples using FIFO
            sample_to_evict = self.uuids.pop()
            self.data_store.remove(sample_to_evict)
        return self.data_store.insert(sample)
