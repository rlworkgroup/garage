from garage.envs.mujoco.sawyer import PickAndPlaceEnv
import numpy as np

env = PickAndPlaceEnv()

for _ in range(1000):
    env.render()
    d = env.sim.data
    # print('----')
    # for coni in range(d.ncon):
    #     print('  Contact %d:' % (coni,))
    #     con = d.contact[coni]
    #     print('    dist     = %0.3f' % (con.dist,))
    #     print('    dim      = %d' % (con.dim,))
    #     print('    geom1    = %d' % (con.geom1,))
    #     print('    geom2    = %d' % (con.geom2,))
    #     # print('    table    = %d, %d' % (id0, id1))
    #     # print('    gripper  = %d %d %d' % (id2, id3, id4))

# #
# id0 = env.sim.model.geom_name2id('table_mesh')
# id1 = env.sim.model.geom_name2id('table_geom')
# id2 = env.sim.model.geom_name2id('first_grip')
# id3 = env.sim.model.geom_name2id('first_grip')
# print('    table    = %d, %d' % (id0, id1))
# print('    gripper  = %d %d' % (id2, id3))
# id4 = env.sim.model.geom_name2id('gripper_geom_finger_r')

id = env.sim.model.geom_name2id('object0')

print(id)

for _ in range(2000):
    env.render()
    action = np.array([0.7, 0.2, 0.3])
    env.sim.data.set_mocap_pos('mocap', action)
    env.sim.data.set_mocap_quat('mocap', np.array([0, 1, 1, 0]))
    env.sim.step()
    # d = env.sim.data
    # print('----')
    # for coni in range(d.ncon):
    #     print('  Contact %d:' % (coni,))
    #     con = d.contact[coni]
    #     print('    dist     = %0.3f' % (con.dist,))
    #     print('    dim      = %d' % (con.dim,))
    #     print('    geom1    = %d' % (con.geom1,))
    #     # print('    geom2    = %d' % (con.geom2,))
    #     # print('    table    = %d, %d' % (id0, id1))
    #     # print('    gripper  = %d %d %d' % (id2, id3, id4))

print('Testing gripper...')
while True:
    env.render()
    action = np.array([0, 0, 0, 20])
    env.step(action)

print('asdas')
for _ in range(20):
    env.render()
    action = np.array([0, 0, -0.1, 10])
    _, _, done, _ = env.step(action)
    # print(env._grasp())
    if done:
        print('...')
        exit()

# print('afih')
# for _ in range(4000):
#     env.render()
#     action = np.array([0, 0, 0, -10])
#     env.step(action)
#     print(env._grasp())
#
# print('uhbfsd')
# for _ in range(5):
#     env.render()
#     action = np.array([0, 0, 0.1, -10])
#     env.step(action)
# print(env._grasp())
