"""Reacher environment for the sawyer robot."""

from gym.spaces import Box
import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco import MujocoEnv
from garage.misc.overrides import overrides


class ReachEnv(MujocoEnv, Serializable):
    """Reacher Environment."""

    FILE = "reach.xml"

    def __init__(self,
                 initial_goal,
                 initial_qpos,
                 distance_threshold=0.05,
                 target_range=0.15,
                 sparse_reward=True,
                 *args,
                 **kwargs):
        """
        Reacher Environment.

        :param initial_goal: initial position to reach.
        :param initial_qpos: initial qpos for each joint.
        :param distance_threshold: distance threhold to define reached.
        :param target_range: delta range the goal is randomized
        :param sparse_reward: whether using sparse reward
        :param args
        :param kwargs
        """
        Serializable.quick_init(self, locals())
        self._initial_goal = initial_goal
        self._distance_threshold = distance_threshold
        self._target_range = target_range
        self._sparse_reward = sparse_reward

        self._goal = self._initial_goal
        super(ReachEnv, self).__init__(*args, **kwargs)
        self._env_setup(initial_qpos)

    @overrides
    def step(self, action):
        """
        Perform one step with action.

        :param action: the action to be performed
        :return: next_obs, reward, done, info
        """
        self.forward_dynamics(action)

        obs = self.get_current_obs()
        next_obs = obs['observation']
        achieved_goal = obs['achieved_goal']
        goal = obs['desired_goal']
        reward = self._compute_reward(achieved_goal, goal)
        done = (self._goal_distance(achieved_goal, goal) <
                self._distance_threshold)

        return Step(next_obs, reward, done)

    def _compute_reward(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = self._goal_distance(achieved_goal, goal)
        if self._sparse_reward:
            return -(d > self._distance_threshold).astype(np.float32)
        else:
            return -d

    @overrides
    def get_current_obs(self):
        """
        Get the current observation.

        :return: current observation.
        """
        grip_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        achieved_goal = np.squeeze(grip_pos.copy())

        obs = np.concatenate([
            grip_pos,
            grip_velp,
            qpos,
            qvel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self._goal
        }

    @staticmethod
    def _goal_distance(goal_a, goal_b):
        """
        Calculate the distance between two goals.

        :param goal_a: first goal.
        :param goal_b: second goal.
        :return: distance between goal a and b.
        """
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def sample_goal(self):
        """
        Sample goals.

        :return: the new sampled goal.
        """
        goal = self._initial_goal.copy()

        random_goal_delta = np.random.uniform(
            -self._target_range, self._target_range, size=2)
        goal[:2] += random_goal_delta
        self._goal = goal
        return goal

    @overrides
    @property
    def observation_space(self):
        """
        Return a Space object.

        :return: observation space
        """
        return Box(
            -np.inf, np.inf, shape=self.get_current_obs()['observation'].shape)

    @overrides
    def close(self):
        """Close the viewer."""
        if self.viewer is not None:
            self.viewer = None
