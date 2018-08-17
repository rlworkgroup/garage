import numpy as np

from garage.envs.mujoco.sawyer import SimplePushEnv

env = SimplePushEnv(
    control_method='task_space_control',
    action_scale=0.04,
    completion_bonus=0.0,
)

low = np.array([-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124])
high = np.array([3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124])

traj = None

start_obs = env.reset()
next_step = start_obs[:3]

for _ in range(50):
    next_step = np.array([0.55, 0, 0.35])
    # next_pos = pos + env.action_space.sample()
    # next_pos_clip = np.clip(next_pos, low, high)
    new_joints = np.array([
        env.env.sim.data.get_joint_qpos('right_j{}'.format(i))
        for i in range(7)
    ])
    if traj is None:
        traj = new_joints
    else:
        traj = np.vstack((traj, new_joints))

    obs, r, done, info = env.step(next_step)
    env.render()

print(traj[-1])
print(traj.shape)

while True:
    # for _ in range(100):
    #     env.env.sim.step()
    env.render()

    # ipdb.set_trace()
