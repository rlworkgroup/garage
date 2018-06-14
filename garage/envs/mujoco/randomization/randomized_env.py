import os.path as osp

import gym
from mujoco_py import load_model_from_xml
from mujoco_py import MjSim

from garage.core import Serializable
from garage.envs.mujoco.mujoco_env import MODEL_DIR


class RandomizedEnv(gym.Env, Serializable):
    """
    This class is just a wrapper class for the MujocoEnv to perform
    the training using Dynamics Randomization.
    Only code in the methods reset and terminate has been added.
    """

    def __init__(self, mujoco_env, variations):
        """
        Set variations with the node in the XML file at file_path.
        """
        Serializable.quick_init(self, locals())
        self._wrapped_env = mujoco_env
        self._variations = variations
        self._file_path = osp.join(MODEL_DIR, mujoco_env.FILE)
        self._variations.initialize_variations(self._file_path)

    def reset(self):
        """
        The new model with randomized parameters is requested and the
        corresponding parameters in the MuJoCo environment class are
        set.
        """
        self._wrapped_env.model = load_model_from_xml(
            self._variations.get_randomized_xml_model())
        if hasattr(self._wrapped_env, 'action_space'):
            del self._wrapped_env.__dict__['action_space']
        self._wrapped_env.sim = MjSim(self._wrapped_env.model)
        self._wrapped_env.data = self._wrapped_env.sim.data
        self._wrapped_env.init_qpos = self._wrapped_env.sim.data.qpos
        self._wrapped_env.init_qvel = self._wrapped_env.sim.data.qvel
        self._wrapped_env.init_qacc = self._wrapped_env.sim.data.qacc
        self._wrapped_env.init_ctrl = self._wrapped_env.sim.data.ctrl
        return self._wrapped_env.reset()

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    def get_param_values(self):
        return self._wrapped_env.get_param_values()

    def set_param_values(self, params):
        self._wrapped_env.set_param_values(params)

    def close(self):
        self._wrapped_env.close()

    @property
    def wrapped_env(self):
        return self._wrapped_env

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    @property
    def horizon(self):
        return self._wrapped_env.horizon


randomize = RandomizedEnv
