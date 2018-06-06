import gym
from rllab.envs.normalized_gym_env import NormalizedGymEnv


def test_env(env):
    rewards = []
    for i in range(100):
        env.reset()
        for e in range(1000):
            # env.render()
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break
    env.close()

    print(env._reward_mean)


env = NormalizedGymEnv(
    gym.make('CartPole-v0'),
    normalize_reward=True,
    normalize_obs=True
)
test_env(env)

env = NormalizedGymEnv(
    gym.make('Blackjack-v0'),
    normalize_obs=True,
    normalize_reward=True,
)
test_env(env)