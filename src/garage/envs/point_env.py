"""Simple 2D environment containing a point and a goal location."""
import gym
import numpy as np

from garage.envs.step import Step


class PointEnv(gym.Env):
    """A simple 2D point environment.

    Attributes:
        observation_space (gym.spaces.Box): The observation space
        action_space (gym.spaces.Box): The action space

    Args:
        goal (np.ndarray): A 2D array representing the goal position
        arena_size (float): The size of arena where the point is constrained
            within (-arena_size, arena_size) in each dimension
        done_bonus (float): A numerical bonus added to the reward
            once the point as reached the goal
        never_done (bool): Never send a `done` signal, even if the
            agent achieves the goal

    """

    def __init__(
            self,
            goal=np.array((1., 1.), dtype=np.float32),
            arena_size=5.,
            done_bonus=0.,
            never_done=False,
    ):
        goal = np.array(goal, dtype=np.float32)
        self._goal = goal
        self._done_bonus = done_bonus
        self._never_done = never_done
        self._arena_size = arena_size

        assert ((goal >= -arena_size) & (goal <= arena_size)).all()

        self._point = np.zeros_like(self._goal)
        self._task = {'goal': self._goal}

    @property
    def observation_space(self):
        """gym.spaces.Box: The observation space."""
        return gym.spaces.Box(low=-np.inf,
                              high=np.inf,
                              shape=(2, ),
                              dtype=np.float32)

    @property
    def action_space(self):
        """gym.spaces.Box: The action space."""
        return gym.spaces.Box(low=-0.1,
                              high=0.1,
                              shape=(2, ),
                              dtype=np.float32)

    def reset(self):
        """Reset the environment.

        Returns:
            np.ndarray: Observation of the environment.

        """
        self._point = np.zeros_like(self._goal)
        return np.copy(self._point)

    def step(self, action):
        """Step the environment state.

        Args:
            action (np.ndarray): The action to take in the environment.

        Returns:
            np.ndarray: Observation. The observation of the environment.
            float: Reward. The reward acquired at this time step.
            boolean: Done. Whether the environment was completed at this
                time step. Always False for this environment.

        """
        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        a = np.clip(a, self.action_space.low, self.action_space.high)

        self._point = np.clip(self._point + a, -self._arena_size,
                              self._arena_size)
        dist = np.linalg.norm(self._point - self._goal)
        done = dist < np.linalg.norm(self.action_space.low)

        # dense reward
        reward = -dist
        # done bonus
        if done:
            reward += self._done_bonus

        # sometimes we don't want to terminate
        done = done and not self._never_done

        return Step(np.copy(self._point), reward, done, task=self._task)

    def render(self, mode='human'):
        """Draw the environment.

        Not implemented.

        Args:
            mode (str): Ignored.

        """
        # pylint: disable=no-self-use

    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, np.ndarray]]: A list of "tasks", where each task is
                a dictionary containing a single key, "goal", mapping to a
                point in 2D space.

        """
        goals = np.random.uniform(-2, 2, size=(num_tasks, 2))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, np.ndarray]): A task (a dictionary containing a
                single key, "goal", which should be a point in 2D space).

        """
        self._task = task
        self._goal = task['goal']
