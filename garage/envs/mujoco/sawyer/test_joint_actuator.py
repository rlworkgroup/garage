
from garage.envs.mujoco.sawyer.pusher_env import PusherEnv

env = PusherEnv(goal_position=(0.2, 0.2, 0.2), control_method="position_control")

for _ in range(30):
    env.reset()
    for _ in range(2000):
        env.render()
        action = env.action_space.sample()
        env.step(action)