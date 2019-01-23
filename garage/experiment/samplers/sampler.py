import abc


class Sampler(abc.ABC):
    """Base Sampler class"""

    @abc.abstractmethod
    def reset(self):
        """Reset environment(s) managed by sampler.

        Returns:
            obs: Tensor(env_n, observation_dim)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, actions):
        """Feed environments with actions.

        Args:
            actions: Tensor(env_n, action_dim)

        Returns: Tensor(env_n, observation_dim)
             Observations under given actions.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_samples(self):
        """Get all samples in current batch.

        Returns: [Path]
            Path: {
                observations: Tensor(path_len, observation_dim),
                actions: Tensor(path_len, action_dim),
                rewards: Tensor(path_len),
                infos: Array(path_len)
            }

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def sample_count(self):
        """Count of interactions with environment in current batch.

        Returns: int

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def path_count(self):
        """Count of completed paths in current batch.

        Returns: int

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_summary(self):
        """Report summary of current batch.

        Returns:
            dict: statistics of current batch.

        """
        raise NotImplementedError
