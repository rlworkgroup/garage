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
        """gym.Env step function."""
        return self.env.step(action)

    def reset(self, **kwargs):
        """gym.Env reset function."""
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            obs = self.env.reset(**kwargs)
        return obs
