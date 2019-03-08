from tests.fixtures.envs.dummy import DummyBoxEnv


class DummyRewardBoxEnv(DummyBoxEnv):
    """A dummy box environment."""

    def __init__(self, random=True):
        super().__init__(random)

    def step(self, action):
        """Step the environment."""
        if action == 0:
            reward = 10
        else:
            reward = -10
        return self.observation_space.sample(), reward, True, dict()
