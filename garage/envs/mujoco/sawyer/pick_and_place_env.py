from gym.envs.robotics import rotations
from gym.spaces import Box
import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco import MujocoEnv
from garage.misc.overrides import overrides


class PickAndPlaceEnv(MujocoEnv, Serializable):

    FILE = 'pick_and_place.xml'

    def __init__(self,
                 initial_goal,
                 distance_threshold=0.05,
                 target_range=0.15,
                 sparse_reward=True,
                 *args,
                 **kwargs):
        Serializable.__init__(self, *args, **kwargs)
        self._initial_goal = initial_goal
        self._distance_threshold = distance_threshold
        self._target_range = target_range
        self._sparse_reward = sparse_reward

        self._goal = self._initial_goal
        super(PickAndPlaceEnv, self).__init__(*args, **kwargs)

    @overrides
    def step(self, action):
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

    def sample_goal(self):
        """
        Sample goals
        :return: the new sampled goal
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
        Returns a Space object
        """
        return Box(
            -np.inf, np.inf, shape=self.get_current_obs()['observation'].shape)

    @overrides
    def get_current_obs(self):
        grip_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        object_pos = self.sim.data.get_site_xpos('object0')
        object_rot = rotations.mat2euler(
            self.sim.data.get_site_xmat('object0'))
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp

        achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([
            grip_pos,
            object_pos.ravel(),
            object_rel_pos.ravel(),
            object_rot.ravel(),
            object_velp.ravel(),
            object_velr.ravel(),
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
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
