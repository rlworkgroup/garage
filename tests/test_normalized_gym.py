import gym

from garage.envs import normalize
from garage.envs.util import flat_dim, flatten



def test_flatten():
    env = normalize(
        gym.make('Pendulum-v0'),
        normalize_reward=True,
        normalize_obs=True,
        flatten_obs=True)
    for i in range(10):
        env.reset()
        for e in range(100):
            env.render()
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            assert next_obs.shape == flat_dim(
                env.observation_space)
            if done:
                break
    env.close()


def test_unflatten():
    env = normalize(
        gym.make('Blackjack-v0'),
        normalize_reward=True,
        normalize_obs=True,
        flatten_obs=False)
    for i in range(10):
        env.reset()
        for e in range(100):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            assert flatten(env.observation_space,
                                     next_obs).shape == flat_dim(
                                         env.observation_space)
            if done:
                break
    env.close()


test_flatten()
test_unflatten()
