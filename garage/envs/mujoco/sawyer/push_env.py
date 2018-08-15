import numpy as np

from gym.spaces import Box
from gym.envs.robotics.utils import reset_mocap2body_xpos

from garage.core.serializable import Serializable
from garage.envs.mujoco.sawyer.sawyer_env import Configuration
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnv
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnvWrapper
from garage.misc.overrides import overrides

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
    ("head", "r_gripper_l_finger_tip"),
    ("head", "r_gripper_r_finger"),
    ("head", "r_gripper_r_finger_tip"),
    ("head", "r_gripper_l_finger"),

    # Close but fine
    ("right_l0", "right_l4_2"),
    ("right_l4_2", "right_l1_2"),
    ("right_l2_2", "pedestal_table"),

    # Trivially okay below this line
    ("r_gripper_l_finger_tip", "r_gripper_r_finger_tip"),
    ("r_gripper_l_finger_tip", "r_gripper_r_finger"),
    ("r_gripper_r_finger_tip", "r_gripper_l_finger"),
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
    ("object0", "r_gripper_r_finger_tip"),
    ("object0", "r_gripper_l_finger"),
    ("object0", "r_gripper_l_finger_tip"),
    ("object0", "pedestal_table"),
    ("mocap", "right_l0"),
    ("mocap", "right_l1"),
    ("mocap", "right_l1_2"),
    ("mocap", "right_l2"),
    ("mocap", "right_l2_2"),
    ("mocap", "right_l3"),
    ("mocap", "right_l4"),
    ('mocap', 'right_l4_2'),
    ("mocap", "right_l5"),
    ("mocap", "right_l6"),
    ("mocap", "right_gripper_base"),
    ("mocap", "right_hand"),
    ("mocap", "r_gripper_r_finger"),
    ("mocap", "r_gripper_r_finger_tip"),
    ("mocap", "r_gripper_l_finger"),
    ("mocap", "r_gripper_l_finger_tip"),
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
    ("achieved_goal", "r_gripper_r_finger_tip"),
    ("achieved_goal", "r_gripper_l_finger"),
    ("achieved_goal", "r_gripper_l_finger_tip"),
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
    ("desired_goal", "r_gripper_r_finger_tip"),
    ("desired_goal", "r_gripper_l_finger"),
    ("desired_goal", "r_gripper_l_finger_tip"),
]


class PushEnv(SawyerEnv):
    def __init__(self,
                 direction="up",
                 easy_gripper_init=True,
                 randomize_start_pos=False,
                 control_method='task_space_control',
                 **kwargs):
        def start_goal_config():
            # center = self.sim.data.get_geom_xpos('target2')
            if randomize_start_pos:
                xy = [
                    np.random.uniform(0.6, 0.8),
                    np.random.uniform(-0.35, 0.35)
                ]
            else:
                xy = [0.7, 0.]
            d = 0.15
            delta = np.array({
                "up": (d, 0),
                "down": (-d, 0),
                "left": (0, d),
                "right": (0, -d)
            }[direction])
            if easy_gripper_init:
                # position gripper besides the block
                gripper_pos = np.concatenate([xy - delta, [0.07]])
            else:
                # position gripper above the block
                gripper_pos = np.concatenate([xy, [0.2]])
            if control_method == 'task_space_control':
                start = Configuration(
                    gripper_pos=gripper_pos,
                    gripper_state=0,
                    object_grasped=False,
                    object_pos=np.concatenate([xy, [0.03]]),
                    joint_pos=None)
                goal = Configuration(
                    gripper_pos=None,
                    gripper_state=0,
                    object_grasped=False,
                    object_pos=np.concatenate([xy + delta, [0.03]]),
                    joint_pos=None)
            else:
                if easy_gripper_init:
                    jpos = np.array({
                        "up": [
                            -0.68198394, -0.96920825, 0.76964638, 2.00488611,
                            -0.56956307, 0.76115281, -0.97169329
                        ],
                        #                        "up": [-0.64455559, -1.0961024,   0.91690344,  2.31425867, -0.57141069,  0.62862147,
                        # -0.69098976],
                        "down": [
                            -0.12526904, 0.29675812, 0.06034621, -0.55948609,
                            -0.03694355, 1.8277617, -1.54921871
                        ],
                        "left": [
                            -0.36766702, 0.62033507, 0.00376033, -1.33212273,
                            0.06092402, 2.29230268, -1.7248123
                        ],
                        "right": [
                            5.97299145e-03, 6.46604393e-01, 1.40055632e-03,
                            -1.22810430e+00, 9.04236294e-03, 2.13193649e+00,
                            -1.38572576e+00
                        ]
                    }[direction])
                else:
                    jpos = np.array([
                        -0.35807692, 0.6890401, -0.21887338, -1.4569705,
                        0.22947722, 2.31383609, -1.4571502
                    ])
                start = Configuration(
                    gripper_pos=gripper_pos,
                    gripper_state=0,
                    object_grasped=False,
                    object_pos=np.concatenate([xy, [0.03]]),
                    joint_pos=jpos)
                goal = Configuration(
                    gripper_pos=None,
                    gripper_state=0,
                    object_grasped=False,
                    object_pos=np.concatenate([xy + delta, [0.03]]),
                    joint_pos=None)
            return start, goal

        def achieved_goal_fn(env: SawyerEnv):
            return env.object_position

        def desired_goal_fn(env: SawyerEnv):
            return env._goal_configuration.object_pos

        self._touched = False

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
            obs = np.concatenate((self.joint_positions, object_pos, object_ori,
                                  gripper_pos))
        else:
            obs = np.concatenate([gripper_pos, object_pos, object_ori])

        achieved_goal = self._achieved_goal_fn(self)
        desired_goal = self._desired_goal_fn(self)

        achieved_goal_qpos = np.concatenate((achieved_goal, [1, 0, 0, 0]))
        self.sim.data.set_joint_qpos('achieved_goal:joint', achieved_goal_qpos)
        desired_goal_qpos = np.concatenate((desired_goal, [1, 0, 0, 0]))
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
            old_gripper_pos = self.gripper_position
            self.joint_positions = next_pos
            self.sim.forward()

            # Move the block
            # Verify if gripper is in collision with block
            in_collision = False
            finger_collision_bodies = [
                'r_gripper_r_finger', 'r_gripper_r_finger_tip',
                'r_gripper_l_finger', 'r_gripper_l_finger_tip'
            ]
            collision_names = self._get_collision_names(whitelist=False)
            for collision_pair in collision_names:
                if 'object0' == collision_pair[0] or 'object0' == collision_pair[1]:
                    if collision_pair[0] in finger_collision_bodies or collision_pair[1] in finger_collision_bodies:
                        in_collision = True
                        break
            if in_collision:
                self._touched = True
                new_gripper_pos = self.gripper_position

                xy_delta = new_gripper_pos[:2] - old_gripper_pos[:2]

                qpos = self.sim.data.get_joint_qpos('object0:joint')
                qpos[0] += xy_delta[0]
                qpos[1] += xy_delta[1]
                self.sim.data.set_joint_qpos('object0:joint', qpos)

                self.sim.forward()

            # self.sim.step()
            # for _ in range(2):
            #     self.sim.step()

            # Verify the execution of the action.
            # for i in range(7):
            #     curr_pos = self.sim.data.get_joint_qpos('right_j{}'.format(i))
            #     d = np.absolute(curr_pos - next_pos[i])
            #     assert d < 1e-2, \
            #         "Joint right_j{} failed to reach the desired qpos.\nError: {}\t Desired: {}\t Current: {}" \
            #         .format(i, d, next_pos[i], curr_pos)
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

        r = self.compute_reward(
            achieved_goal=obs.get('achieved_goal'),
            desired_goal=obs.get('desired_goal'),
            info=info)

        if not self._touched:
            r += self.compute_reward(
                self.gripper_position, self.object_position, info=info)

        if self._easy_gripper_init:
            # encourage gripper to move close to block
            r += 0.02 - np.linalg.norm(self.gripper_position -
                                       self.object_position)

        self._is_success = self._success_fn(self, self._achieved_goal,
                                            self._desired_goal, info)
        done = False
        if self._is_success:
            r = self._completion_bonus
            done = True

        info["r"] = r
        info["d"] = done

        # control cost
        r -= self._control_cost_coeff * np.linalg.norm(a)

        return obs, r, done, info

    @overrides
    def reset(self):
        self._touched = False
        return super(PushEnv, self).reset()


class SimplePushEnv(SawyerEnvWrapper, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.reward_range = None
        self.metadata = None
        super().__init__(PushEnv(*args, **kwargs))
