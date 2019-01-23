import abc


class Policy(abc.ABC):
    @abc.abstractmethod
    def sample(self, obs):
        """
        Sample actions given observations.

        Args:
            obs: Tensor(batch_size, observation_dim)
                 Observations.

        Returns: (Tensor(batch_size, action_dim), Tensor(batch_size))
                 Actions and their log probability.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def logpdf(self, obs, action):
        """
        Compute action log probability given observation.

        Args:
            obs: Tensor(batch_size, observation_dim)
            action: Tensor(batch_size, action_dim)

        Returns: Tensor(batch_size)
                 Log probability of action under current policy

        """
        raise NotImplementedError
