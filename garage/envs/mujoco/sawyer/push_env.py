import numpy as np

from gym.spaces import Box
from gym.envs.robotics.utils import reset_mocap2body_xpos

from garage.core.serializable import Serializable
from garage.envs.mujoco.sawyer.sawyer_env import Configuration
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnv
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnvWrapper
from garage.misc.overrides import overrides
from gym.envs.robotics import rotations

COLLISION_WHITELIST = [
    # Liberal whitelist here
    # Remove this section for a more conservative policy

    # The head seems to have a large collision body
    ("head", "right_l0"),
    ("head", "right_l1"),
    ("head", "right_l1_2"),
    ("head", "right_l2"),
    ("head", "right_l2_2"),
    ("head", "right_l3"),
    ("head", "right_l4"),
    ("head", "right_l4_2"),
    ("head", "right_l5"),
    ("head", "right_l6"),
    ("head", "right_gripper_base"),
    # ("head", "r_gripper_l_finger_tip"),
    ("head", "r_gripper_r_finger"),
    # ("head", "r_gripper_r_finger_tip"),
    ("head", "r_gripper_l_finger"),

    # Close but fine
    ("right_l0", "right_l4_2"),
    ("right_l4_2", "right_l1_2"),
    ("right_l2_2", "pedestal_table"),

    # Trivially okay below this line
    # ("r_gripper_l_finger_tip", "r_gripper_r_finger_tip"),
    # ("r_gripper_l_finger_tip", "r_gripper_r_finger"),
    # ("r_gripper_r_finger_tip", "r_gripper_l_finger"),
    # ("task_marker", "right_l0"),
    # ("task_marker", "right_l1"),
    # ("task_marker", "right_l1_2"),
    # ("task_marker", "right_l2"),
    # ("task_marker", "right_l2_2"),
    # ("task_marker", "right_l3"),
    # ("task_marker", "right_l4"),
    # ('task_marker', 'right_l4_2'),
    # ("task_marker", "right_l5"),
    # ("task_marker", "right_l6"),
    # ("task_marker", "right_gripper_base"),
    # ("task_marker", "right_hand"),
    # ("task_marker", "r_gripper_r_finger"),
    # ("task_marker", "r_gripper_r_finger_tip"),
    # ("task_marker", "r_gripper_l_finger"),
    # ("task_marker", "r_gripper_l_finger_tip"),
    ("object0", "right_l0"),
    ("object0", "right_l1"),
    ("object0", "right_l1_2"),
    ("object0", "right_l2"),
    ("object0", "right_l2_2"),
    ("object0", "right_l3"),
    ("object0", "right_l4"),
    ("object0", "right_l4_2"),
    ("object0", "right_l5"),
    ("object0", "right_l6"),
    ("object0", "right_gripper_base"),
    ("object0", "right_hand"),
    ("object0", "r_gripper_r_finger"),
    # ("object0", "r_gripper_r_finger_tip"),
    ("object0", "r_gripper_l_finger"),
    # ("object0", "r_gripper_l_finger_tip"),
    ("object0", "pedestal_table"),
    # ("mocap", "right_l0"),
    # ("mocap", "right_l1"),
    # ("mocap", "right_l1_2"),
    # ("mocap", "right_l2"),
    # ("mocap", "right_l2_2"),
    # ("mocap", "right_l3"),
    # ("mocap", "right_l4"),
    # ('mocap', 'right_l4_2'),
    # ("mocap", "right_l5"),
    # ("mocap", "right_l6"),
    # ("mocap", "right_gripper_base"),
    # ("mocap", "right_hand"),
    # ("mocap", "r_gripper_r_finger"),
    # ("mocap", "r_gripper_r_finger_tip"),
    # ("mocap", "r_gripper_l_finger"),
    # ("mocap", "r_gripper_l_finger_tip"),
    ("achieved_goal", "right_l0"),
    ("achieved_goal", "right_l1"),
    ("achieved_goal", "right_l1_2"),
    ("achieved_goal", "right_l2"),
    ("achieved_goal", "right_l2_2"),
    ("achieved_goal", "right_l3"),
    ("achieved_goal", "right_l4"),
    ("achieved_goal", "right_l4_2"),
    ("achieved_goal", "right_l5"),
    ("achieved_goal", "right_l6"),
    ("achieved_goal", "right_gripper_base"),
    ("achieved_goal", "right_hand"),
    ("achieved_goal", "r_gripper_r_finger"),
    # ("achieved_goal", "r_gripper_r_finger_tip"),
    ("achieved_goal", "r_gripper_l_finger"),
    # ("achieved_goal", "r_gripper_l_finger_tip"),
    ("desired_goal", "right_l0"),
    ("desired_goal", "right_l1"),
    ("desired_goal", "right_l1_2"),
    ("desired_goal", "right_l2"),
    ("desired_goal", "right_l2_2"),
    ("desired_goal", "right_l3"),
    ("desired_goal", "right_l4"),
    ("desired_goal", "right_l4_2"),
    ("desired_goal", "right_l5"),
    ("desired_goal", "right_l6"),
    ("desired_goal", "right_gripper_base"),
    ("desired_goal", "right_hand"),
    ("desired_goal", "r_gripper_r_finger"),
    # ("desired_goal", "r_gripper_r_finger_tip"),
    ("desired_goal", "r_gripper_l_finger"),
    # ("desired_goal", "r_gripper_l_finger_tip"),
]


class PushEnv(SawyerEnv):
    def __init__(self,
                 delta=np.array([0.15, 0, 0]),
                 easy_gripper_init=True,
                 randomize_start_pos=False,
                 control_method='task_space_control',
                 **kwargs):
        self._already_successful = False

        def start_goal_config():
            # center = self.sim.data.get_geom_xpos('target2')
            if randomize_start_pos:
                initial_block_pos = np.array([
                    np.random.uniform(0.6, 0.8),
                    np.random.uniform(-0.35, 0.35),
                    0.065
                ])
            else:
                initial_block_pos = np.array([0.64, 0.22, 0.065])
            # d = 0.15
            # delta = np.array({
            #     "up": (d, 0),
            #     "down": (-d, 0),
            #     "left": (0, d),
            #     "right": (0, -d)
            # }[direction])
            # if easy_gripper_init:
            #     # position gripper besides the block
            #     gripper_pos = np.concatenate([initial_block_pos - delta, [0.07]])
            # else:
            #     # position gripper above the block
            #     gripper_pos = np.concatenate([initial_block_pos, [0.2]])
            if control_method == 'task_space_control':
                start = Configuration(
                    gripper_pos=initial_block_pos + np.array([0, 0, 0.3]),
                    gripper_state=0,
                    object_grasped=False,
                    object_pos=initial_block_pos,
                    joint_pos=None)
                goal = Configuration(
                    gripper_pos=None,
                    gripper_state=0,
                    object_grasped=False,
                    object_pos=initial_block_pos + delta,
                    joint_pos=None)
            else:
                if easy_gripper_init:
                    # jpos = np.array([
                    #     0.02631256,  0.57778916,  0.1339495, -2.16678053,
                    #     -1.77062755, 3.03287272, -3.22155594
                    # ])
                    # jpos = np.array([-0.7318902, -1.06648196, 0.92428461, 1.78847105, -0.43512962, 1.08968813,
                    #         -1.05157125])
 #                    jpos = np.array([-0.83591221, -1.05890433, 0.91351439, 1.75259073, -0.43152266, 1.11151082,
 # -1.1771668])
 #                    jpos = np.array([0, -1.1, 0, 1.3, 0, 1.4, 1.65])
                    jpos = np.array([-0.4839443359375, -0.991173828125, -2.3821015625, -1.9510517578125, -0.5477119140625, -0.816458984375, -0.816326171875])
                    # jpos = np.array({
                    #     "up": [
                    #         -0.68198394, -0.96920825, 0.76964638, 2.00488611,
                    #         -0.56956307, 0.76115281, -0.97169329
                    #     ],
                    #      "up": [-0.64455559, -3.0961024,   0.91690344,  4.31425867, -0.57141069,  0.62862147,
                    #     -0.69098976],
                    #     "up": [
                    #         -0.140923828125, -1.2789248046875, -3.043166015625,
                    #         -2.139623046875, -0.047607421875, -0.7052822265625,
                    #         -1.4102060546875
                    #     ],
                    #     "down": [
                    #         -0.12526904, 0.29675812, 0.06034621, -0.55948609,
                    #         -0.03694355, 1.8277617, -1.54921871
                    #     ],
                    #     "down": [
                    #         -0.140923828125, -1.2789248046875, -3.043166015625,
                    #         -2.139623046875, -0.047607421875, -0.7052822265625,
                    #         -1.4102060546875
                    #     ],
                    #     "left": [
                    #         -0.36766702, 0.62033507, 0.00376033, -1.33212273,
                    #         0.06092402, 2.29230268, -1.7248123
                    #     ],
                    #     "right": [
                    #         5.97299145e-03, 6.46604393e-01, 1.40055632e-03,
                    #         -1.22810430e+00, 9.04236294e-03, 2.13193649e+00,
                    #         -1.38572576e+00
                    #     ]
                    # }[direction])
                else:
                    jpos = np.array([
                        -0.35807692, 0.6890401, -0.21887338, -1.4569705,
                        0.22947722, 2.31383609, -1.4571502
                    ])
                start = Configuration(
                    gripper_pos=initial_block_pos + np.array([0, 0, 0.3]),
                    gripper_state=0,
                    object_grasped=False,
                    object_pos=initial_block_pos,
                    joint_pos=jpos)
                goal = Configuration(
                    gripper_pos=None,
                    gripper_state=0,
                    object_grasped=False,
                    object_pos=initial_block_pos + delta,
                    joint_pos=None)
            return start, goal

        def achieved_goal_fn(env: SawyerEnv):
            return env.object_position

        def desired_goal_fn(env: SawyerEnv):
            return env._goal_configuration.object_pos

        self._test_ration = 0

        # depends on block's size
        # 1----------2
        # |          |
        # |          |
        # 4----------3
        length = 0.11
        width = 0.11
        self.cp1 = np.array([- width / 2, - length / 2, 0])
        self.cp2 = np.array([- width / 2, length / 2, 0])
        self.cp3 = np.array([width / 2, length / 2, 0])
        self.cp4 = np.array([width / 2, - length / 2, 0])

        super(PushEnv, self).__init__(
            start_goal_config=start_goal_config,
            achieved_goal_fn=achieved_goal_fn,
            desired_goal_fn=desired_goal_fn,
            file_path="push.xml",
            collision_whitelist=COLLISION_WHITELIST,
            control_method=control_method,
            **kwargs)
        self._easy_gripper_init = easy_gripper_init

    def get_obs(self):
        gripper_pos = self.gripper_position
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        object_pos = self.object_position
        object_ori = self.object_orientation
        grasped = self.has_object
        if self._control_method == 'position_control':
            # obs = np.concatenate((self.joint_positions, object_pos, object_ori,
            #                       gripper_pos))
            # relative difference
            initial_jpos = np.array(
                [-0.4839443359375, -0.991173828125, -2.3821015625, -1.9510517578125, -0.5477119140625, -0.816458984375,
                 -0.816326171875])
            obs = np.concatenate((self.joint_positions - initial_jpos, gripper_pos - object_pos, object_ori-np.array([1., 0., 0., 0.])))
            print(obs)
        else:
            obs = np.concatenate([gripper_pos, object_pos, object_ori])

        achieved_goal = self._achieved_goal_fn(self)
        desired_goal = self._desired_goal_fn(self)

        achieved_goal_qpos = np.concatenate((achieved_goal, [1, 0, 0, 0]))
        self.sim.data.set_joint_qpos('achieved_goal:joint', achieved_goal_qpos)
        desired_goal_qpos = np.concatenate((np.array(desired_goal).reshape(-1), [1, 0, 0, 0]))
        self.sim.data.set_joint_qpos('desired_goal:joint', desired_goal_qpos)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
            'gripper_state': self.gripper_state,
            'gripper_pos': gripper_pos.copy(),
            'has_object': grasped,
            'object_pos': object_pos.copy()
        }

    @overrides
    @property
    def action_space(self):
        if self._control_method == 'torque_control':
            return super(SawyerEnv, self).action_space
        elif self._control_method == 'task_space_control':
            return Box(
                np.array([-0.15, -0.15, -0.15]),
                np.array([0.15, 0.15, 0.15]),
                dtype=np.float32)
        elif self._control_method == 'position_control':
            return Box(
                low=np.full(7, -0.04), high=np.full(7, 0.04), dtype=np.float32)
        else:
            raise NotImplementedError

    @overrides
    def step(self, action):
        # Clip to action space
        assert action.shape == self.action_space.shape
        a = action.copy()  # Note: you MUST copy the action if you modify it
        a *= self._action_scale
        a = np.clip(a, self.action_space.low, self.action_space.high)

        if self._control_method == "torque_control":
            self.forward_dynamics(a)
            self.sim.forward()
        elif self._control_method == "task_space_control":
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_pos[0, :
                                    3] = self.sim.data.mocap_pos[0, :3] + a[:3]
            self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
            # self.set_gripper_state(a[3])
            self.sim.forward()
            for _ in range(1):
                self.sim.step()
        elif self._control_method == "position_control":
            curr_pos = self.joint_positions
            next_pos = np.clip(a + curr_pos, self.joint_position_space.low,
                               self.joint_position_space.high)
            old_block_pos = self.object_position
            old_block_ori = self.object_orientation
            old_finger_pos = self.finger_position
            self.joint_positions = next_pos
            self.sim.forward()

            # Move the block
            # Verify if gripper is in collision with block
            in_collision = False
            finger_collision_bodies = [
                'r_gripper_r_finger', # 'r_gripper_r_finger_tip',
                'r_gripper_l_finger', # 'r_gripper_l_finger_tip'
            ]
            collision_names = self._get_collision_names(whitelist=False)
            for collision_pair in collision_names:
                if 'object0' == collision_pair[0] or 'object0' == collision_pair[1]:
                    if collision_pair[0] in finger_collision_bodies or collision_pair[1] in finger_collision_bodies:
                        in_collision = True
                        break
            if in_collision:
                new_finger_pos = self.finger_position
                if self.finger_position[2] >= self.object_position[2]:
                    if not self.in_xyregion(old_finger_pos, old_block_pos, old_block_ori):
                        # Make sure the gripper is not pulling the block
                        delta_finger_block = old_block_pos - old_finger_pos
                        delta_finger = new_finger_pos - old_finger_pos
                        if np.dot(delta_finger_block, delta_finger) > 0:

                            xy_delta = new_finger_pos[:2] - old_finger_pos[:2]

                            qpos = self.sim.data.get_joint_qpos('object0:joint')
                            qpos[0] += xy_delta[0]
                            qpos[1] += xy_delta[1]
                            self.sim.data.set_joint_qpos('object0:joint', qpos)

                            self.sim.forward()
                        else:
                            self.go_back()
                    else:
                        self.go_back()
                else:
                    self.go_back()

            self.previous_joint_positions = self.joint_positions
        else:
            raise NotImplementedError

        # force object's velocity to 0
        # self.sim.data.set_joint_qvel('object0:joint', np.zeros(6))
        # self.sim.data.qacc[:6] = 0
        self._step += 1

        obs = self.get_obs()

        self._achieved_goal = obs.get('achieved_goal')
        self._desired_goal = obs.get('desired_goal')

        info = {
            "l": self._step,
            "grasped": obs["has_object"],
            "gripper_state": obs["gripper_state"],
            "gripper_position": obs["gripper_pos"],
            "object_position": obs["object_pos"],
            "is_success": self._is_success
        }

        desired_goal = obs.get('desired_goal')
        achieved_goal = obs.get('achieved_goal')
        revert_unit_vec = - (desired_goal - achieved_goal) / np.linalg.norm(desired_goal - achieved_goal)
        block_desired_gripper = revert_unit_vec * 0.09 + achieved_goal

        r4 = 0
        r1 = self.compute_reward(
            achieved_goal=obs.get('achieved_goal'),
            desired_goal=obs.get('desired_goal'),
            info=info) * 2

        r2 = -np.linalg.norm(self.finger_position - block_desired_gripper) / 3

        # w, x, y, z quat
        upright_gripper = np.array([0, 0, 1, 0])
        gripper_rot = rotations.mat2quat(self.sim.data.get_site_xmat('grip'))
        r3 = -np.linalg.norm(upright_gripper - gripper_rot) / 10 * 2/3

        # if r2 / r1 > self._test_ration:
        #     self._test_ration = r2 / r1
        #     print(self._test_ration)

        end_position = self.object_position + np.array([0, 0, 0.15])
        if self._already_successful:
            r2 = -np.linalg.norm(self.finger_position - end_position) / 5 * 2/3
        r = r1 + r2 + r3 + r4

        # if self._easy_gripper_init:
        #     # encourage gripper to move close to block
        #     r += 0.02 - np.linalg.norm(self.gripper_position -
        #                                self.object_position)

        self._is_success = self._success_fn(self, self._achieved_goal,
                                            self._desired_goal, info)
        done = False
        if self._is_success and not self._already_successful:
            self._already_successful = True
            r = self._completion_bonus
            done = False

        info["r"] = r
        info["d"] = done

        # control cost
        r -= self._control_cost_coeff * np.linalg.norm(a)

        return obs, r, done, info

    def go_back(self):
        self.joint_positions = self.previous_joint_positions
        self.sim.forward()

    @overrides
    def reset(self):
        self._already_successful = False

        return super(PushEnv, self).reset()

        self.previous_joint_positions = self.joint_positions

    def in_xyregion(self, gripper_pos, block_pos, block_ori):
        p = np.array([gripper_pos[0], gripper_pos[1], 0])
        newcp1 = rotate(block_ori, self.cp1)
        newcp2 = rotate(block_ori, self.cp2)
        newcp3 = rotate(block_ori, self.cp3)
        newcp4 = rotate(block_ori, self.cp4)

        newc = np.array([block_pos[0], block_pos[1], 0])
        newp1 = newcp1 + newc
        newp2 = newcp2 + newc
        newp3 = newcp3 + newc
        newp4 = newcp4 + newc

        p1p2 = newp2 - newp1
        p1p = p - newp1

        p3p4 = newp4 - newp3
        p3p = p - newp3

        p2p3 = newp3 - newp2
        p2p = p - newp2

        p4p1 = newp1 - newp4
        p4p = p - newp4

        return np.dot(np.cross(p1p2, p1p), np.cross(p3p4, p3p)) >= 0 and np.dot(np.cross(p2p3, p2p), np.cross(p4p1, p4p)) >= 0


def rotate(ori, vec):
    w = ori[0]
    x = ori[1]
    y = ori[2]
    z = ori[3]

    rotation_matrix = np.matrix([[1-2*y*y-2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                                 [2*x*y + 2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                                 [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])

    new_vec = rotation_matrix * np.matrix([[vec[0]], [vec[1]], [vec[2]]])

    return np.array([new_vec[0, 0], new_vec[1, 0], new_vec[2, 0]])


class SimplePushEnv(SawyerEnvWrapper, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.reward_range = None
        self.metadata = None
        super().__init__(PushEnv(*args, **kwargs))
