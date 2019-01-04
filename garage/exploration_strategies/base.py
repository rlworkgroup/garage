class ExplorationStrategy:
    def get_action(self, t, observation, policy, sess=None, **kwargs):
        raise NotImplementedError

    def get_actions(self, t, observations, policy, sess=None, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass
