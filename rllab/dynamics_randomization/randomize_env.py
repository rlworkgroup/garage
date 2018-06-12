import os.path as osp

from mujoco_py import MjSim

from rllab.core import Serializable
from rllab.dynamics_randomization.mujoco_model_gen import MujocoModelGenerator
from rllab.envs import Env
from rllab.envs.mujoco.mujoco_env import MODEL_DIR


class RandomizedEnv(Env, Serializable):
    def __init__(self, mujoco_env, variations):
        Serializable.quick_init(self, locals())
        self._wrapped_env = mujoco_env
        self._variations = variations
        self._file_path = osp.join(MODEL_DIR, mujoco_env.FILE)

        self._mujoco_model = MujocoModelGenerator(self._file_path, variations)

    def reset(self):
        self._wrapped_env.model = self._mujoco_model.get_model()
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

    def terminate(self):
        self._wrapped_env.terminate()

    def get_param_values(self):
        return self._wrapped_env.get_param_values()

    def set_param_values(self, params):
        self._wrapped_env.set_param_values(params)

    def terminate(self):
        self._mujoco_model.stop()

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
