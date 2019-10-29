from garage.tf.policies import Policy


class UniformControlPolicy(Policy):

    def __init__(
            self,
            env_spec,
    ):
        super(UniformControlPolicy, self).__init__(env_spec=env_spec)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        return self.action_space.sample(), dict()

    def get_actions(self, observations):
        return self.action_space.sample_n(len(observations)), dict()

    def get_params_internal(self, **tags):
        return []
