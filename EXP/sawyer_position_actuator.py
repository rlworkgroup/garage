from garage.envs.mujoco.sawyer import PickAndPlaceEnv
import numpy as np

env = PickAndPlaceEnv()

env.reset()

# Go down
for _ in range(2000):
    env.render()
    action = np.array([0.9, 0.20, 0.2])
    env.sim.data.set_mocap_pos('mocap', action)
    env.sim.data.set_mocap_quat('mocap', np.array([0, 1, 1, 0]))
    env.sim.step()

# Open the gripper and go down
for _ in range(220):
    env.render()
    action = np.array([0, 0, -0.03, 0.005])
    env.step(action)

# Close the gripper and go up
for _ in range(500):
    env.render()
    action = np.array([0, 0, 0.05, -0.005])
    env.step(action)
