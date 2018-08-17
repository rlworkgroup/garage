import numpy as np

from garage.envs.mujoco.sawyer import SimplePushEnv


env = SimplePushEnv(
    control_method='position_control',
    action_scale=0.04,
    completion_bonus=0.0,
)


gripper_pos = [0.04, 0.06, 5]
block_pos = [0, 0, 5]
block_ori = [0.707, 0, 0, 0.707]

if env.env.in_xyregion(gripper_pos, block_pos, block_ori):
    print('In xy region!')
