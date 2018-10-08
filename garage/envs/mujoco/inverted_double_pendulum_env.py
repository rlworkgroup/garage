import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco.mujoco_env import MujocoEnv
from garage.misc import autoargs
from garage.misc.overrides import overrides


class InvertedDoublePendulumEnv(MujocoEnv, Serializable):
    FILE = 'inverted_double_pendulum.xml.mako'

    @autoargs.arg(
        "random_start",
        type=bool,
        help="Randomized starting position by adjusting the angles"
        "When this is false, the double pendulum started out"
        "in balanced position")
    def __init__(self, random_start=True, *args, **kwargs):
        self.random_start = random_start
        super().__init__(*args, **kwargs)

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).reshape(-1)

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x**2 + (y - 2)**2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = float(alive_bonus - dist_penalty - vel_penalty)
        done = y <= 1
        return Step(next_obs, r, done)

    @overrides
    def reset_mujoco(self, init_state=None):
        assert init_state is None
        qpos = np.copy(self.init_qpos)
        if self.random_start:
            qpos[1] = (np.random.rand() - 0.5) * 40 / 180. * np.pi
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = self.init_qvel
        self.sim.data.qacc[:] = self.init_qacc
        self.sim.data.ctrl[:] = self.init_ctrl
