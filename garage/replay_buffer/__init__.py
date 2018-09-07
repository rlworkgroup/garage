"""
This public package contains the replay buffer primitives.

The replay buffer primitives can be used for RL algorithms.
"""
from garage.replay_buffer.her_replay_buffer import HerReplayBuffer
from garage.replay_buffer.regular_replay_buffer import RegularReplayBuffer

__all__ = ["HerReplayBuffer", "RegularReplayBuffer"]
