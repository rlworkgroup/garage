import gym
import numpy as np

from garage.envs.base import Step


class PointEnv(gym.Env):
    """A simple 2D point environment.

    Attributes:
        observation_space (:obj:`gym.spaces.Box`): The observation space
        action_space (:obj:`gym.spaces.Box`): The action space

    Args:
        goal (:obj:`np.ndarray`, optional): A 2D array representing the goal
            position
        done_bonus (float, optional): A numerical bonus added to the reward
            once the point as reached the goal
        never_done (bool, optional): Never send a `done` signal, even if the
            agent achieves the goal.
    """

    def __init__(
            self,
            goal=np.array((1., 1.), dtype=np.float32),
            done_bonus=0.,
            never_done=False,
    ):
        self._goal = np.array(goal, dtype=np.float32)
        self._done_bonus = done_bonus
        self._never_done = never_done

        self._point = np.zeros_like(self._goal)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf,
                              high=np.inf,
                              shape=(2, ),
                              dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(low=-0.1,
                              high=0.1,
                              shape=(2, ),
                              dtype=np.float32)

    def reset(self):
        self._point = np.zeros_like(self._goal)
        return np.copy(self._point)

    def step(self, action):
        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        a = np.clip(a, self.action_space.low, self.action_space.high)

        dist = np.linalg.norm(self._point - self._goal)
        done = dist < np.linalg.norm(self.action_space.low)

        # dense reward
        reward = -dist
        # done bonus
        if done:
            reward += self._done_bonus

        # sometimes we don't want to terminate
        done = done and not self._never_done

        return Step(np.copy(self._point), reward, done)

    def render(self, mode='human'):
        pass
