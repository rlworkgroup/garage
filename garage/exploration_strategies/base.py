class ExplorationStrategy:
    def get_action(self, t, observation, policy, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass
