class ExplorationStrategy:
    def get_action(self, t, observation, policy, **kwargs):
        raise NotImplementedError

    def get_actions(self, t, observations, policy, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass
