import tempfile
import os
import os.path as osp
import warnings

from cached_property import cached_property
import mako.template
import mako.lookup
import mujoco_py
from mujoco_py import functions
from mujoco_py import load_model_from_path
from mujoco_py import MjSim
from mujoco_py import MjViewer
import numpy as np
import theano

from rllab import spaces
from rllab.envs import Env
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc import logger

warnings.simplefilter(action='ignore', category=FutureWarning)

MODEL_DIR = osp.abspath(
    osp.join(osp.dirname(__file__), '../../../vendor/mujoco_models'))

BIG = 1e6


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


class MujocoEnv(Env):
    FILE = None

    @autoargs.arg(
        'action_noise',
        type=float,
        help='Noise added to the controls, which will be '
        'proportional to the action bounds')
    def __init__(self, action_noise=0.0, file_path=None, template_args=None):
        # compile template
        if file_path is None:
            if self.__class__.FILE is None:
                raise "Mujoco file not specified"
            file_path = osp.join(MODEL_DIR, self.__class__.FILE)
        if file_path.endswith(".mako"):
            lookup = mako.lookup.TemplateLookup(directories=[MODEL_DIR])
            with open(file_path) as template_file:
                template = mako.template.Template(
                    template_file.read(), lookup=lookup)
            content = template.render(
                opts=template_args if template_args is not None else {}, )
            tmp_f, file_path = tempfile.mkstemp(text=True)
            with open(file_path, 'w') as f:
                f.write(content)
            self.model = load_model_from_path(file_path)
            os.close(tmp_f)
        else:
            self.model = load_model_from_path(file_path)
        self.sim = MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self.init_qpos = self.sim.data.qpos
        self.init_qvel = self.sim.data.qvel
        self.init_qacc = self.sim.data.qacc
        self.init_ctrl = self.sim.data.ctrl
        self.qpos_dim = self.init_qpos.size
        self.qvel_dim = self.init_qvel.size
        self.ctrl_dim = self.init_ctrl.size
        self.action_noise = action_noise
        self.frame_skip = 1
        self.dcom = None
        self.current_com = None
        self.reset()
        super(MujocoEnv, self).__init__()

    @cached_property
    @overrides
    def action_space(self):
        bounds = self.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return spaces.Box(lb, ub)

    @cached_property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @property
    def action_bounds(self):
        return self.action_space.bounds

    def reset_mujoco(self, init_state=None):
        self.sim.reset()
        if init_state is None:
            self.sim.data.qpos[:] = self.init_qpos + \
                                   np.random.normal(size=self.init_qpos.shape) * 0.01
            self.sim.data.qvel[:] = self.init_qvel + \
                                   np.random.normal(size=self.init_qvel.shape) * 0.1
            self.sim.data.qacc[:] = self.init_qacc
            self.sim.data.ctrl[:] = self.init_ctrl
        else:
            start = 0
            for datum_name in ["qpos", "qvel", "qacc", "ctrl"]:
                datum = getattr(self.sim.data, datum_name)
                datum_dim = datum.shape[0]
                datum = init_state[start:start + datum_dim]
                setattr(self.sim.data, datum_name, datum)
                start += datum_dim

    @overrides
    def reset(self, init_state=None):
        self.reset_mujoco(init_state)
        self.sim.forward()
        self.current_com = self.sim.data.subtree_com[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    def get_current_obs(self):
        return self._get_full_obs()

    def _get_full_obs(self):
        data = self.sim.data
        cdists = np.copy(self.sim.geom_margin).flat
        for c in self.sim.data.contact:
            cdists[c.geom2] = min(cdists[c.geom2], c.dist)
        obs = np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            data.cinert.flat,
            data.cvel.flat,
            data.qfrc_actuator.flat,
            data.cfrc_ext.flat,
            data.qfrc_constraint.flat,
            cdists,
            self.dcom.flat,
        ])
        return obs

    @property
    def _state(self):
        return np.concatenate(
            [self.sim.data.qpos.flat, self.sim.data.qvel.flat])

    @property
    def _full_state(self):
        return np.concatenate([
            self.sim.data.qpos,
            self.sim.data.qvel,
            self.sim.data.qacc,
            self.sim.data.ctrl,
        ]).ravel()

    def inject_action_noise(self, action):
        # generate action noise
        noise = self.action_noise * \
                np.random.normal(size=action.shape)
        # rescale the noise to make it proportional to the action bounds
        lb, ub = self.action_bounds
        noise = 0.5 * (ub - lb) * noise
        return action + noise

    def forward_dynamics(self, action):
        self.sim.data.ctrl[:] = self.inject_action_noise(action)
        for _ in range(self.frame_skip):
            self.sim.step()
        self.sim.forward()
        new_com = self.sim.data.subtree_com[0]
        self.dcom = new_com - self.current_com
        self.current_com = new_com

    def get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
        return self.viewer

    def render(self, close=False, mode='human'):
        if mode == 'human':
            viewer = self.get_viewer()
            viewer.render()
        elif mode == 'rgb_array':
            viewer = self.get_viewer()
            viewer.render()
            data, width, height = viewer.get_image()
            return np.fromstring(
                data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        if close:
            self.stop_viewer()

    def start_viewer(self):
        viewer = self.get_viewer()
        if not viewer.running:
            viewer.start()

    def stop_viewer(self):
        if self.viewer:
            self.viewer.finish()

    def release(self):
        # temporarily alleviate the issue (but still some leak)
        functions.mj_deleteModel(self.sim._wrapped)
        functions.mj_deleteData(self.data._wrapped)

    def get_body_xmat(self, body_name):
        return self.data.get_body_xmat(body_name).reshape((3, 3))

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def get_body_comvel(self, body_name):
        return self.data.get_body_xvelp(body_name)

    def print_stats(self):
        super(MujocoEnv, self).print_stats()
        print("qpos dim:\t%d" % len(self.sim.data.qpos))

    def action_from_key(self, key):
        raise NotImplementedError
