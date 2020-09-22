"""Fire reset wrapper for gym.Env."""
import gym


class FireReset(gym.Wrapper):
    """Fire reset wrapper for gym.Env.

    Take action "fire" on reset.

    Args:
        env (gym.Env): The environment to be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE', (
            'Only use fire reset wrapper for suitable environment!')
        assert len(env.unwrapped.get_action_meanings()) >= 3, (
            'Only use fire reset wrapper for suitable environment!')

    def step(self, action):
        """gym.Env step function.

        Args:
            action (int): index of the action to take.

        Returns:
            np.ndarray: Observation conforming to observation_space
            float: Reward for this step
            bool: Termination signal
            dict: Extra information from the environment.
        """
        return self.env.step(action)

    def reset(self, **kwargs):
        """gym.Env reset function.

        Args:
            kwargs (dict): extra arguments passed to gym.Env.reset()

        Returns:
            np.ndarray: next observation.
        """
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs
