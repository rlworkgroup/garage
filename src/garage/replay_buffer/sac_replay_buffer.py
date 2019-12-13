import numpy as np


class SACReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, env_spec, max_size):
        obs_dim = env_spec.observation_space.shape[0]
        act_dim = env_spec.action_space.shape[0]
        self.obs1_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([max_size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.n_transitions_stored, self.max_size = 0, 0, max_size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.n_transitions_stored = min(self.n_transitions_stored+1, self.max_size)
    
    def add_transitions(self, **kwargs):
        """
        Add multiple transitions into the replay buffer.

        A transition contains one or multiple entries, e.g.
        observation, action, reward, terminal and next_observation.
        The same entry of all the transitions are stacked, e.g.
        {'observation': [obs1, obs2, obs3]} where obs1 is one
        numpy.ndarray observation from the environment.

        Args:
            kwargs (dict(str, [numpy.ndarray])): Dictionary that holds
                the transitions.
        """
        assert(len(kwargs["observation"]) == 1)
        self.store(kwargs["observation"][0], kwargs["action"][0], kwargs["reward"],  kwargs["next_observation"][0], kwargs["terminal"])

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.n_transitions_stored, size=batch_size)
        return dict(observation=self.obs1_buf[idxs],
                    next_observation=self.obs2_buf[idxs],
                    action=self.acts_buf[idxs],
                    reward=self.rews_buf[idxs],
                    terminal=self.done_buf[idxs])