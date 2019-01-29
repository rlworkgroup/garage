import gym

from garage.core import Serializable


class IdentificationEnv(gym.Wrapper, Serializable):
    def __init__(self, mdp_cls, mdp_args):
        self.mdp_cls = mdp_cls
        self.mdp_args = dict(mdp_args)
        # Leaving this commented out so that tests can pass. It will be
        # removable (along with this class, possibly) as soon as we move out
        # the garage.envs.box2d and garage.envs.mujoco
        # See https://github.com/rlworkgroup/garage/issues/359
        # self.mdp_args["template_args"] = dict(noise=True)
        mdp = self.gen_mdp()
        super().__init__(mdp)

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def gen_mdp(self):
        return self.mdp_cls(**self.mdp_args)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.close()
        self.env = self.gen_mdp()
        return self.env.reset()
