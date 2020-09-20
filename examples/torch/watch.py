from garage.experiment import Snapshotter
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from garage.envs import GymEnv
import numpy as np
from garage import rollout

snapshotter = Snapshotter()
data = snapshotter.load('/home/mishari/dqn_benchmarks/Qbert/QbertNoFrameskip-v4_1')
myenv = data['env']
exploration_policy = data['algo'].exploration_policy
exploration_policy.policy._qf.to('cpu'),
env = wrap_deepmind(make_atari('QbertNoFrameskip-v4'), frame_stack=True)
env = GymEnv(env)

print(env.observation_space.shape)
print(myenv.observation_space.shape)
# See what the trained policy can accomplish
ep_rewards = np.asarray([])
# for i in range(10):
#     path = rollout(env, exploration_policy, animated=True, speedup=0.2)
#     ep_rewards = np.append(ep_rewards, np.sum(path['rewards']))

# print("AVG REWARD {}".format(np.mean(ep_rewards)))


ep_rewards = np.asarray([])
for i in range(10):
    path = rollout(myenv, exploration_policy.policy, animated=True, speedup=0.2)
    ep_rewards = np.append(ep_rewards, np.sum(path['rewards']))

print("AVG REWARD {}".format(np.mean(ep_rewards)))

policy = data['algo'].policy
policy._qf.to('cpu'),
env = data['env']
