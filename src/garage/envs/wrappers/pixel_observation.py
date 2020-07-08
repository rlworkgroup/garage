"""Pixel observation wrapper for gym.Env."""
# yapf: disable
import gym
from gym.wrappers.pixel_observation import (
    PixelObservationWrapper as gymWrapper)

# yapf: enable


class PixelObservationWrapper(gym.Wrapper):
    """Pixel observation wrapper for obtaining pixel observations.

    Instead of returning the default environment observation, the wrapped
    environment's render function is used to produce RGB pixel observations.

    This behaves like gym.wrappers.PixelObservationWrapper but returns a
    gym.spaces.Box observation space and observation instead of
    a gym.spaces.Dict.

    Args:
        env (gym.Env): The environment to wrap. This environment must produce
            non-pixel observations and have a Box observation space.
        headless (bool): If true, this creates a window to init GLFW. Set to
            true if running on a headless machine or with a dummy X server,
            false otherwise.

    """

    def __init__(self, env, headless=True):
        if headless:
            # pylint: disable=import-outside-toplevel
            # this import fails without a valid mujoco license
            # so keep this here to avoid unecessarily requiring
            # a mujoco license everytime the wrappers package is
            # accessed.
            from mujoco_py import GlfwContext
            GlfwContext(offscreen=True)
        env.reset()
        env = gymWrapper(env)
        super().__init__(env)
        self._observation_space = env.observation_space['pixels']

    @property
    def observation_space(self):
        """gym.spaces.Box: Environment observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def reset(self, **kwargs):
        """gym.Env reset function.

        Args:
            kwargs (dict): Keyword arguments to be passed to gym.Env.reset.

        Returns:
            np.ndarray: Pixel observation of shape :math:`(O*, )`
                from the wrapped environment.
        """
        return self.env.reset(**kwargs)['pixels']

    def step(self, action):
        """gym.Env step function.

        Performs one action step in the enviornment.

        Args:
            action (np.ndarray): Action of shape :math:`(A*, )`
                to pass to the environment.

        Returns:
            np.ndarray: Pixel observation of shape :math:`(O*, )`
                from the wrapped environment.
            float : Amount of reward returned after previous action.
            bool : Whether the episode has ended, in which case further step()
                calls will return undefined results.
            dict: Contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning).
        """
        obs, reward, done, info = self.env.step(action)
        return obs['pixels'], reward, done, info
