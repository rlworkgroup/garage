"""Simple 2D environment containing a point and a goal location."""
import akro
import math
import numpy as np

from garage import Environment, TimeStep, StepType
from garage.envs import EnvSpec

class PointEnv(Environment):
    """A simple 2D point environment.

    Args:
        goal (np.ndarray): A 2D array representing the goal position
        arena_size (float): The size of arena where the point is constrained
            within (-arena_size, arena_size) in each dimension
        done_bonus (float): A numerical bonus added to the reward
            once the point as reached the goal
        never_done (bool): Never send a `done` signal, even if the
            agent achieves the goal
         max_episode_length (int): The maximum steps allowed for an episode.

    """

    def __init__(
        self,
        goal=np.array((1., 1.), dtype=np.float32),
        arena_size=5.,
        done_bonus=0.,
        never_done=False,
        max_episode_length=math.inf
    ):
        goal = np.array(goal, dtype=np.float32)
        self._goal = goal
        self._done_bonus = done_bonus
        self._never_done = never_done
        self._arena_size = arena_size

        assert ((goal >= -arena_size) & (goal <= arena_size)).all()

        self._last_observation = None
        self._step_cnt = 0
        self._max_episode_length = max_episode_length

        self._point = np.zeros_like(self._goal)
        self._task = {'goal': self._goal}
        self._observation_space = akro.Box(low=-np.inf,
                                                 high=np.inf,
                                                 shape=(3, ),
                                                 dtype=np.float32)
        self._action_space = akro.Box(low=-0.1,
                                            high=0.1,
                                            shape=(2, ),
                                            dtype=np.float32)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return []

    def reset(self):
        """Call reset on wrapped env.

        Returns:
            numpy.ndarray: The first observation. It must conforms to
            `observation_space`.
            dict: The episode-level information. Note that this is not part
            of `env_info` provided in `step()`. It contains information of
            the entire episode， which could be needed to determine the first
            action (e.g. in the case of goal-conditioned or MTRL.)

        """
        self._point = np.zeros_like(self._goal)
        dist = np.linalg.norm(self._point - self._goal)

        first_obs = np.concatenate([self._point, (dist, )])
        self._last_observation = first_obs
        self._step_cnt = 0
        episode_info = {}
        return first_obs, episode_info

    def step(self, action):
        """Call step on wrapped env.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            TimeStep: The time step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment
            has been
                constructed and `reset()` has not been called.

        """
        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        a = np.clip(a, self.action_space.low, self.action_space.high)

        self._point = np.clip(self._point + a, -self._arena_size,
                              self._arena_size)
        dist = np.linalg.norm(self._point - self._goal)
        succ = dist < np.linalg.norm(self.action_space.low)

        # dense reward
        reward = -dist
        # done bonus
        if succ:
            reward += self._done_bonus
        # Type conversion
        if not isinstance(reward, float):
            reward = float(reward)

        # sometimes we don't want to terminate
        done = succ and not self._never_done

        obs = np.concatenate([self._point, (dist, )])

        last_obs = self._last_observation
        self._last_observation = last_obs
        self._step_cnt += 1

        step_type = None
        if done:
            step_type = StepType.TERMINAL
        elif self._step_cnt >= self._spec.max_episode_length:
            step_type = StepType.TIMEOUT
        elif self._step_cnt == 1:
            step_type = StepType.FIRST
        else:
            step_type = StepType.MID

        return TimeStep(
            env_spec=self.spec,
            observation=last_obs,
            action=action,
            reward=reward,
            next_observation=obs,
            env_info={'task': self._task, 'success': succ},
            agent_info={},  # TODO: can't be populated by env
            step_type=step_type)

    def render(self, mode):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.
        """
        pass

    def visualize(self):
        """Creates a visualization of the environment."""
        pass

    def close(self):
        """Close the wrapped env."""
        pass

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

