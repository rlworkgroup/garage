from rllab.core import Parameterized
from rllab.core import Serializable
from rllab.distributions import Delta
from rllab.policies import Policy
from rllab.misc import overrides


class UniformControlPolicy(Policy):
    def __init__(
            self,
            env_spec,
    ):
        Serializable.quick_init(self, locals())
        super(UniformControlPolicy, self).__init__(env_spec=env_spec)

    @overrides
    def get_action(self, observation):
        return self.action_space.sample(), dict()

    def get_params_internal(self, **tags):
        return []

    def get_actions(self, observations):
        return self.action_space.sample_n(len(observations)), dict()

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        pass

    @property
    def distribution(self):
        # Just a placeholder
        return Delta()
