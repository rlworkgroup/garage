import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnv
from garage.misc.overrides import overrides


class BlockStackingEnv(SawyerEnv, Serializable):

    FILE = 'block_stacking.xml'

    def __init__(self, block_size=0.025, *args, **kwargs):
        Serializable.__init__(self, *args, **kwargs)
        super(BlockStackingEnv, self).__init__(
            initial_goal=None, initial_qpos=None, *args, **kwargs)
        self._distance_threshold = block_size / 2.
        self._done = False

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        obs = self.get_current_obs()
        reward = self.compute_reward()
        done = self._done

        return Step(obs, reward, done)

    @overrides
    def compute_reward(self, achieved_goal, desired_goal, info):
        green_pos = self.sim.data.get_geom_xpos('object0')
        red_pos = self.sim.data.get_geom_xpos('object1')
        blue_pos = self.sim.data.get_geom_xpos('object2')

        if self._is_stacked(green_pos, red_pos) and self._is_stacked(
                green_pos, blue_pos) and self._is_stacked(red_pos, blue_pos):
            self._done = True
            return 100
        else:
            return -1

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

    def _is_stacked(self, com_1, com_2):
        distance = np.sqrt((com_1[0] - com_2[0])**2 + (com_1[1] - com_2[1])**2)
        return distance < self._distance_threshold

    @overrides
    def reset(self, init_state=None):
        self._done = False
        return super(BlockStackingEnv, self).reset(init_state)
