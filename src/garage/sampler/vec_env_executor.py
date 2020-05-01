"""Environment wrapper that runs multiple environments."""
import warnings

import numpy as np

from garage.misc import tensor_utils


class VecEnvExecutor:
    """Environment wrapper that runs multiple environments.

    Args:
        envs (list[gym.Env]): List of environments to batch together.
        max_path_length (int): Maximum length of any path.

    """

    def __init__(self, envs, max_path_length):
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length

        warnings.warn(
            DeprecationWarning(
                'VecEnvExecutor is deprecated, and will be removed in the '
                'next release. Please use VecWorker and one of the new '
                'samplers which implement garage.sampler.Sampler, such as '
                'RaySampler'))

    def step(self, action_n):
        """Step all environments using the provided actions.

        Inserts an environment infor 'vec_env_executor.complete' containing the
        episode end signal (time limit reached or done signal from
        environment).

        Args:
            action_n (np.ndarray): Array of actions.

        Returns:
            tuple: Tuple containing:
                * observations (np.ndarray)
                * rewards (np.ndarray)
                * dones (np.ndarray): The done signal from the environment.
                * env_infos (dict[str, np.ndarray])

        """
        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rewards, dones, env_infos = list(
            map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        completes = np.asarray(dones)
        if self.max_path_length is not None:
            completes[self.ts >= self.max_path_length] = True
        for (i, complete) in enumerate(completes):
            if complete:
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
            env_infos[i]['vec_env_executor.complete'] = completes
        return (obs, rewards, dones,
                tensor_utils.stack_tensor_dict_list(env_infos))

    def reset(self):
        """Reset all environments.

        Returns:
            np.ndarray: Observations of shape :math:`(K, O*)`

        """
        results = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        """Read the number of environments.

        Returns:
            int: Number of environments

        """
        return len(self.envs)

    @property
    def action_space(self):
        """Read the action space.

        Returns:
            gym.Space: The action space.

        """
        return self._action_space

    @property
    def observation_space(self):
        """Read the observation space.

        Returns:
            gym.Space: The observation space.

        """
        return self._observation_space

    def close(self):
        """Close all environments."""
