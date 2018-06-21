import numpy as np

from garage.exploration_strategies import ExplorationStrategy
from garage.misc.overrides import overrides


class OUStrategy(ExplorationStrategy):
    def __init__(self, env_spec, mu=0, sigma=0.3, theta=0.15, dt=1e-2,
                 x0=None):
        self.env_spec = env_spec
        self.action_space = env_spec.action_space
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def simulate(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=len(x))
        self.state = x + dx
        return x

    @overrides
    def reset(self):
        self.state = self.x0 if self.x0 is not None else self.mu * np.ones(
            self.action_space.shape[-1])

    @overrides
    def get_action(self, t, observation, policy, **kwargs):
        action = policy.get_action(observation)
        ou_state = self.simulate()
        return np.clip(action + ou_state, self.action_space.low,
                       self.action_space.high)


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    ou = OUStrategy(
        env_spec=gym.make("Pendulum-v0"), mu=0, theta=0.15, sigma=0.3)
    states = []
    for i in range(1000):
        states.append(ou.simulate()[0])

    plt.plot(states)
    plt.show()
