import math

import mujoco_py  # noqa: F401, pylint: disable=unused-import
import glfw  # noqa: I100
import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco import MujocoEnv
from garage.misc.overrides import overrides


class PointEnv(MujocoEnv, Serializable):
    """
    Use Left, Right, Up, Down, A (steer left), D (steer right)
    """

    FILE = 'point.xml'

    def __init__(self, *args, **kwargs):
        super(PointEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def step(self, action):
        qpos = np.copy(self.sim.data.qpos)
        qpos[2] += action[1]
        ori = qpos[2]
        # compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]
        # ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -7, 7)
        qpos[1] = np.clip(qpos[1] + dy, -7, 7)
        self.sim.data.qpos[:] = qpos
        self.sim.forward()
        next_obs = self.get_current_obs()
        return Step(next_obs, 0, False)

    def get_xy(self):
        qpos = self.sim.data.qpos
        return qpos[0], qpos[1]

    def set_xy(self, xy):
        qpos = np.copy(self.sim.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]
        self.sim.data.qpos[:] = qpos
        self.sim.forward()

    @overrides
    def action_from_key(self, key):
        lb, ub = self.action_bounds
        if key == glfw.KEY_LEFT:
            return np.array([0, ub[0] * 0.3])
        elif key == glfw.KEY_RIGHT:
            return np.array([0, lb[0] * 0.3])
        elif key == glfw.KEY_UP:
            return np.array([ub[1], 0])
        elif key == glfw.KEY_DOWN:
            return np.array([lb[1], 0])
        else:
            return np.array([0, 0])
