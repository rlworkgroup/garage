import os.path as osp

import numpy as np

from garage.envs.mujoco.sawyer import SimplePushEnv

env = SimplePushEnv(
    control_method='position_control',
    action_scale=1.0,
    completion_bonus=0.0,
)

# model = mujoco_py.load_model_from_path('/home/hejia/Projects/garage/vendor/mujoco_models/push.xml')
# sim = mujoco_py.MjSim(model)
# sim.forward()
# mj_viewer = mujoco_py.MjViewer(sim)
#
# sim.data.set_joint_qpos('object0:joint', [0.7, 0, 0.03, 1, 0, 0, 0])

low = np.array([-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124])
high = np.array([3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124])

traj = np.load(osp.join(osp.dirname(__file__), 'traj.npy'))

while True:
    start_obs = env.reset()
    curr_positions = start_obs[:7]

    for joint_step in traj:
        incr_step = joint_step - curr_positions
        obs, r, done, info = env.step(incr_step)
        curr_positions = obs[:7]
        env.render()
