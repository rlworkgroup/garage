import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnv
from garage.misc.overrides import overrides


class BinSortingEnv(SawyerEnv, Serializable):

    FILE = 'bin_sorting.xml'

    def __init__(self,
                 green_bin_center=(0.85, 0.25),
                 red_bin_center=(0.85, 0.55),
                 blue_bin_center=(0.85, -0.05),
                 *args,
                 **kwargs):
        Serializable.__init__(self, *args, **kwargs)
        self._green_bin_center = green_bin_center
        self._red_bin_center = red_bin_center
        self._blue_bin_center = blue_bin_center
        self._bin_radius = 0.08

        self._green_done = False
        self._red_done = False
        self._blue_done = False

        super(BinSortingEnv, self).__init__(
            initial_goal=None, initial_qpos=None, *args, **kwargs)

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        obs = self.get_current_obs()
        reward = self.compute_reward()
        done = (self._green_done and self._red_done and self._blue_done)
        return Step(obs, reward, done)

    @overrides
    def get_current_obs(self):
        grip_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        green_object_pos = self.sim.data.get_geom_xpos('object0')
        red_object_pos = self.sim.data.get_geom_xpos('object1')
        blue_object_pos = self.sim.data.get_geom_xpos('object2')

        obs = np.concatenate([
            grip_pos,
            green_object_pos,
            red_object_pos,
            blue_object_pos,
            grip_velp,
            qpos,
            qvel,
        ])

        return obs

    @overrides
    def compute_reward(self, achieved_goal, desired_goal, info):
        green_object_pos = self.sim.data.get_geom_xpos('object0')
        red_object_pos = self.sim.data.get_geom_xpos('object1')
        blue_object_pos = self.sim.data.get_geom_xpos('object2')

        if self.in_circle(self._green_bin_center, green_object_pos):
            self._green_done = True
        if self.in_circle(self._red_bin_center, red_object_pos):
            self._red_done = True
        if self.in_circle(self._blue_bin_center, blue_object_pos):
            self._blue_done = True

        if self._green_done and self._blue_done and self._red_done:
            return 100
        else:
            return -1

    def in_circle(self, center, obj_pos):
        distance = np.sqrt((obj_pos[0] - center[0])**2 +
                           (obj_pos[1] - center[1])**2)
        return distance < self._bin_radius

    @overrides
    def reset(self, init_state=None):
        self._green_done = False
        self._blue_done = False
        self._red_done = False
        return super(BinSortingEnv, self).reset(init_state)
