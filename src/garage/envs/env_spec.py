from garage.core import Serializable


class EnvSpec(Serializable):
    def __init__(self, observation_space, action_space):
        """
        :type observation_space: Space
        :type action_space: Space
        """
        self.observation_space = observation_space
        self.action_space = action_space

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())
