import gym

from rllab.envs.normalized_gym_env import NormalizedGymEnv, \
    gym_space_flatten_dim, gym_space_flatten


def test_flatten():
    env = NormalizedGymEnv(
        gym.make('Pendulum-v0'),
        normalize_reward=True,
        normalize_obs=True,
        flatten=True)
    for i in range(100):
        env.reset()
        for e in range(100):
            env.render()
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            assert next_obs.shape == gym_space_flatten_dim(
                env.observation_space)
            if done:
                break
    env.close()


def test_unflatten():
    env = NormalizedGymEnv(
        gym.make('Blackjack-v0'),
        normalize_reward=True,
        normalize_obs=True,
        flatten=False)
    for i in range(100):
        env.reset()
        for e in range(100):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            assert gym_space_flatten(env.observation_space,
                                     next_obs).shape == gym_space_flatten_dim(
                                         env.observation_space)
            if done:
                break
    env.close()


test_flatten()
test_unflatten()
