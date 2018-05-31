from lxml import etree
from rllab.envs import Env
import os.path as osp
import numpy as np
from rllab.dynamics_randomization import VariationMethods
from rllab.dynamics_randomization.variation import VariationDistributions
from mujoco_py import load_model_from_xml
from mujoco_py import MjSim
from rllab.core import Serializable

MODEL_DIR = osp.abspath(
    osp.join(osp.dirname(__file__), '../../vendor/mujoco_models'))


class RandomizedEnv(Env, Serializable):
    def __init__(self, mujoco_env, variations):
        Serializable.quick_init(self, locals())
        self._wrapped_env = mujoco_env
        self._variations = variations
        self._file_path = osp.join(MODEL_DIR, mujoco_env.FILE)
        self._model = etree.parse(self._file_path)

        for v in variations.get_list():
            e = self._model.find(v.xpath)
            v.elem = e

            # todo: handle AttributeError
            val = e.attrib[v.attrib].split(' ')
            if len(val) == 1:
                v.default = float(e.attrib[v.attrib])
            else:
                v.default = np.array(list(map(float, val)))

    def reset(self):
        for v in self._variations.get_list():
            e = v.elem
            # todo: handle size
            if v.distribution == VariationDistributions.GAUSSIAN:
                c = np.random.normal(loc=v.var_range[0], scale=v.var_range[1])
            elif v.distribution == VariationDistributions.UNIFORM:
                c = np.random.uniform(low=v.var_range[0], high=v.var_range[1])
            if v.method == VariationMethods.COEFFICIENT:
                e.attrib[v.attrib] = str(c * v.default)
            elif v.method == VariationMethods.ABSOLUTE:
                e.attrib[v.attrib] = str(c)
            else:
                raise NotImplementedError("Unknown method")

        model_xml = etree.tostring(self._model.getroot()).decode("ascii")
        self._wrapped_env.model = load_model_from_xml(model_xml)
        self._wrapped_env.sim = MjSim(self._wrapped_env.model)
        self._wrapped_env.data = self._wrapped_env.sim.data
        self._wrapped_env.viewer = None
        self._wrapped_env.init_qpos = self._wrapped_env.sim.data.qpos
        self._wrapped_env.init_qvel = self._wrapped_env.sim.data.qvel
        self._wrapped_env.init_qacc = self._wrapped_env.sim.data.qacc
        self._wrapped_env.init_ctrl = self._wrapped_env.sim.data.ctrl
        self._wrapped_env.qpos_dim = self._wrapped_env.init_qpos.size
        self._wrapped_env.qvel_dim = self._wrapped_env.init_qvel.size
        self._wrapped_env.ctrl_dim = self._wrapped_env.init_ctrl.size
        self._wrapped_env.frame_skip = 1
        self._wrapped_env.dcom = None
        self._wrapped_env.current_com = None
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
