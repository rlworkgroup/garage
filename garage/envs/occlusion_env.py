from cached_property import cached_property
import gym
import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco import MujocoEnv
from garage.envs.util import flat_dim
from garage.misc.overrides import overrides

BIG = 1e6


class OcclusionEnv(gym.Wrapper, Serializable):
    """ Occludes part of the observation."""

    def __init__(self, env, sensor_idx):
        """
        :param sensor_idx: list or ndarray of indices to be shown.
         Other indices will be occluded. Can be either list of integer
         indices or boolean mask.
        """
        Serializable.quick_init(self, locals())

        self._set_sensor_mask(env, sensor_idx)
        super(OcclusionEnv, self).__init__(env)
        self._dt = 1
        if isinstance(env, MujocoEnv):
            self._dt = env.sim.opt.timestep * env.frame_skip

    def _set_sensor_mask(self, env, sensor_idx):
        obsdim = flat_dim(env.observation_space)
        if len(sensor_idx) > obsdim:
            raise ValueError(("Length of sensor mask ({0}) cannot be greater "
                              "than observation dim ({1})").format(
                                  len(sensor_idx), obsdim))
        if len(sensor_idx) == obsdim and not np.any(np.array(sensor_idx) > 1):
            sensor_mask = np.array(sensor_idx, dtype=np.bool)
        elif np.any(np.unique(sensor_idx, return_counts=True)[1] > 1):
            raise ValueError(("Double entries or boolean mask "
                              "with dim ({0}) < observation dim ({1})").format(
                                  len(sensor_idx), obsdim))
        else:
            sensor_mask = np.zeros((obsdim, ), dtype=np.bool)
            sensor_mask[sensor_idx] = 1
        self._sensor_mask = sensor_mask

    def occlude(self, obs):
        return obs[self._sensor_mask]

    def get_current_obs(self):
        return self.occlude(self.env.get_current_obs())

    @cached_property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return gym.spaces.Box(ub * -1, ub, dtype=np.float32)

    @overrides
    def reset(self):
        obs = self.env.reset()
        return self.occlude(obs)

    @overrides
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(self.occlude(next_obs), reward, done, **info)

    @property
    def dt(self):
        return self._dt
