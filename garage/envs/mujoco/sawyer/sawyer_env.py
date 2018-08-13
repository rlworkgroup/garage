from collections import namedtuple
import os.path as osp

import gym
from gym.envs.robotics import rotations
from gym.envs.robotics.utils import reset_mocap_welds, reset_mocap2body_xpos
from gym.spaces import Box
import numpy as np

from garage.envs.mujoco import MujocoEnv
from garage.envs.mujoco.mujoco_env import MODEL_DIR
from garage.misc.overrides import overrides

# 'head',
# 'screen',
# 'head_camera',
# 'right_torso_itb',
# 'right_arm_itb',
# 'right_hand_camera',
# 'right_wrist',
# 'right_hand',
# 'right_gripper_base',

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

    ("task_marker", "right_l0"),
    ("task_marker", "right_l1"),
    ("task_marker", "right_l1_2"),
    ("task_marker", "right_l2"),
    ("task_marker", "right_l2_2"),
    ("task_marker", "right_l3"),
    ("task_marker", "right_l4"),
    ('task_marker', 'right_l4_2'),
    ("task_marker", "right_l5"),
    ("task_marker", "right_l6"),
    ("task_marker", "right_gripper_base"),
    ("task_marker", "right_hand"),
    ("task_marker", "r_gripper_r_finger"),
    ("task_marker", "r_gripper_r_finger_tip"),
    ("task_marker", "r_gripper_l_finger"),
    ("task_marker", "r_gripper_l_finger_tip"),
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

Configuration = namedtuple(
    "Configuration",
    ["gripper_pos", "gripper_state", "object_grasped", "object_pos"])


def default_reward_fn(env, achieved_goal, desired_goal, _info: dict):
    d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    if env._reward_type == 'sparse':
        return (d < env._distance_threshold).astype(np.float32)
    return -d


def default_success_fn(env, achieved_goal, desired_goal, _info: dict):
    return (np.linalg.norm(achieved_goal - desired_goal, axis=-1) <
            env._distance_threshold).astype(np.float32)


def default_achieved_goal_fn(env):
    return env.gripper_position


def default_desired_goal_fn(env):
    if env._goal_configuration.object_grasped and not env.has_object:
        return env.object_position
    return env._goal_configuration.gripper_pos


class SawyerEnv(MujocoEnv, gym.GoalEnv):
    """Sawyer Robot Environments."""

    def __init__(self,
                 start_goal_config,
                 reward_fn=default_reward_fn,
                 success_fn=default_success_fn,
                 achieved_goal_fn=default_achieved_goal_fn,
                 desired_goal_fn=default_desired_goal_fn,
                 max_episode_steps=50,
                 completion_bonus=10,
                 distance_threshold=0.05,
                 for_her=False,
                 control_cost_coeff=0.,
                 action_scale=1.0,
                 randomize_start_jpos=False,
                 collision_whitelist=COLLISION_WHITELIST,
                 terminate_on_collision=False,
                 collision_penalty=0.,
                 reward_type='dense',
                 control_method='task_space_control',
                 file_path='pick_and_place.xml',
                 *args,
                 **kwargs):
        """
        Sawyer Environment.
        :param args:
        :param kwargs:
        """
        self._start_goal_config = start_goal_config
        self._reward_fn = reward_fn
        self._success_fn = success_fn
        self._achieved_goal_fn = achieved_goal_fn
        self._desired_goal_fn = desired_goal_fn

        self._start_configuration = None  # type: Configuration
        self._goal_configuration = None  # type: Configuration
        self._achieved_goal = None  # type: np.array
        self._desired_goal = None  # type: np.array
        self.gripper_state = 0.
        self._is_success = False

        self._reward_type = reward_type
        self._control_method = control_method
        self._max_episode_steps = max_episode_steps
        self._completion_bonus = completion_bonus
        self._distance_threshold = distance_threshold
        self._step = 0
        self._for_her = for_her
        self._control_cost_coeff = control_cost_coeff
        self._action_scale = action_scale
        self._randomize_start_jpos = randomize_start_jpos
        self._terminate_on_collision = terminate_on_collision
        self._collision_penalty = collision_penalty

        file_path = osp.join(MODEL_DIR, file_path)
        MujocoEnv.__init__(self, file_path=file_path)

        # Populate and id-based whitelist of acceptable body ID contacts
        self._collision_whitelist = []
        for c in collision_whitelist:
            # Hedge our bets by allowing both orderings
            self._collision_whitelist.append((
                self.sim.model.body_name2id(c[0]),
                self.sim.model.body_name2id(c[1])
            ))
            self._collision_whitelist.append((
                self.sim.model.body_name2id(c[1]),
                self.sim.model.body_name2id(c[0])
            ))

        self.env_setup()

    def _sample_start_goal(self):
        if isinstance(self._start_goal_config, tuple):
            self._start_configuration, self._goal_configuration = self._start_goal_config
        else:
            self._start_configuration, self._goal_configuration = self._start_goal_config(
            )

    def env_setup(self):
        reset_mocap_welds(self.sim)
        self.sim.forward()

    @property
    def joint_position_space(self):
        low = np.array(
            [-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124])
        high = np.array(
            [3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124])
        return Box(low, high, dtype=np.float32)

    @property
    def joint_positions(self):
        curr_pos = []
        for i in range(7):
            curr_pos.append(
                self.sim.data.get_joint_qpos('right_j{}'.format(i)))
        return np.array(curr_pos)

    @joint_positions.setter
    def joint_positions(self, jpos):
        for i, p in enumerate(jpos):
            self.sim.data.set_joint_qpos('right_j{}'.format(i), p)

    def set_gripper_position(self, position):
        reset_mocap2body_xpos(self.sim)
        self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
        self.sim.data.set_mocap_pos('mocap', position)
        for _ in range(100):
            self.sim.step()
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
            self.sim.data.set_mocap_pos('mocap', position)
        # self.sim.forward()

    @property
    def gripper_position(self):
        return self.sim.data.get_site_xpos('grip') - np.array(
            [0., 0., .1])  # 0.1 offset for the finger

    def set_object_position(self, position):
        object_qpos = np.concatenate((position, [1, 0, 0, 0]))
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

    @property
    def object_position(self):
        return self.sim.data.get_site_xpos('object0').copy()

    @property
    def has_object(self):
        """Determine if the object is grasped"""
        contacts = tuple()
        for coni in range(self.sim.data.ncon):
            con = self.sim.data.contact[coni]
            contacts += ((con.geom1, con.geom2), )
        finger_id_1 = self.sim.model.geom_name2id('finger_tip_1')
        finger_id_2 = self.sim.model.geom_name2id('finger_tip_2')
        object_id = self.sim.model.geom_name2id('object0')
        if ((finger_id_1, object_id) in contacts or
            (object_id, finger_id_1) in contacts) and (
                (finger_id_2, object_id) in contacts or
                (object_id, finger_id_2) in contacts):
            return True
        else:
            return False

    @overrides
    @property
    def action_space(self):
        if self._control_method == 'torque_control':
            return super(SawyerEnv, self).action_space
        elif self._control_method == 'task_space_control':
            return Box(
                np.array([-0.15, -0.15, -0.15, -1.]),
                np.array([0.15, 0.15, 0.15, 1.]),
                dtype=np.float32)
        elif self._control_method == 'position_control':
            return Box(
                low=np.full(7, -0.04), high=np.full(7, 0.04), dtype=np.float32)
        else:
            raise NotImplementedError

    @overrides
    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.get_obs()['observation'].shape,
            dtype=np.float32)

    def step(self, action):
        assert action.shape == self.action_space.shape

        # Note: you MUST copy the action if you modify it
        a = action.copy()

        # Clip to action space
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
            self.set_gripper_state(a[3])
            for _ in range(5):
                self.sim.step()
            self.sim.forward()
        elif self._control_method == "position_control":
            curr_pos = self.joint_positions
            next_pos = np.clip(
                a + curr_pos,
                self.joint_position_space.low,
                self.joint_position_space.high
            )
            self.joint_positions = next_pos
            self.sim.step()

            # Verify the execution of the action.
            # for i in range(7):
            #     curr_pos = self.joint_positions
            #     d = np.absolute(curr_pos[i] - next_pos[i])
            #     assert d < 1e-5, \
            #     "Joint right_j{} failed to reached the desired qpos.\nError: {}\t Desired: {}\t Current: {}"\
            #     .format(i, d, next_pos[i], curr_pos)

        else:
            raise NotImplementedError
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

        self._is_success = self._success_fn(self, self._achieved_goal,
                                            self._desired_goal, info)
        done = False

        # control cost
        r -= self._control_cost_coeff * np.linalg.norm(a)

        # collision detection
        if self.in_collision:
            r -= self._collision_penalty
            if self._terminate_on_collision:
                done = True

        if self._is_success:
            r = self._completion_bonus
            # done = True

        info["r"] = r
        info["d"] = done

        return obs, r, done, info

    def set_gripper_state(self, state):
        # 1 = open, -1 = closed
        self.gripper_state = state
        state = (state + 1.) / 2.
        self.sim.data.ctrl[:] = np.array([state * 0.020833, -state * 0.020833])
        # for _ in range(3):
        #     self.sim.step()
        # self.sim.forward()

    def get_obs(self):
        gripper_pos = self.gripper_position
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        object_pos = self.object_position
        object_rot = rotations.mat2euler(
            self.sim.data.get_site_xmat('object0'))
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        object_rel_pos = object_pos - gripper_pos
        object_velp -= grip_velp
        grasped = self.has_object
        obs = np.concatenate([
            gripper_pos,
            object_pos.ravel(),  # TODO remove object_pos (reveals task id)
            object_rel_pos.ravel(),
            object_rot.ravel(),
            object_velp.ravel(),
            object_velr.ravel(),
            grip_velp,
            qpos,
            qvel,
            [float(grasped), self.gripper_state],
        ])

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

    def is_success(self):
        return self._is_success

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._reward_fn(self, achieved_goal, desired_goal, info)

    @property
    def in_collision(self):
        for c in self._get_collisions():
            if c not in self._collision_whitelist:
                return True

        return False


    def _get_collision_names(self, whitelist=True):
        contacts = []
        for c in self._get_collisions():
            if c not in self._collision_whitelist or not whitelist:
                contacts.append((
                    self.sim.model.body_id2name(c[0]),
                    self.sim.model.body_id2name(c[1])
                ))
        return contacts

    def _get_collisions(self):
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            if c.geom1 != 0 and c.geom2 !=0:
                yield (
                    self.sim.model.geom_bodyid[c.geom1],
                    self.sim.model.geom_bodyid[c.geom2]
                )

    @overrides
    def reset(self):
        self._step = 0
        super(SawyerEnv, self).reset()

        self._sample_start_goal()
        self.set_object_position(self._start_configuration.object_pos)

        # if self._start_configuration.object_grasped:
        #     self.set_gripper_state(1)  # open
        #     self.set_gripper_position(self._start_configuration.gripper_pos)
        #     self.set_object_position(self._start_configuration.gripper_pos)
        #     self.set_gripper_state(-1)  # close
        # else:
        #     self.set_gripper_state(self._start_configuration.gripper_state)
        #     self.set_gripper_position(self._start_configuration.gripper_pos)
        #     self.set_object_position(self._start_configuration.object_pos)

        # for _ in range(20):
        #     self.sim.step()
        # self.sim.forward()

        attempts = 1
        if self._randomize_start_jpos:
            self.joint_positions = self.joint_position_space.sample()
            self.sim.step()
            while hasattr(self, "_collision_whitelist") and self.in_collision:
                if attempts > 1000:
                    print("Gave up after 1000 attempts")
                    import ipdb
                    ipdb.set_trace()

                self.joint_positions = self.joint_position_space.sample()
                self.sim.step()
                for _ in range(100): self.get_viewer().render()
                attempts += 1



        return self.get_obs()


def ppo_info(info):
    info["task"] = [1]
    ppo_infos = {"episode": info}
    return ppo_infos


class SawyerEnvWrapper:
    def __init__(self,
                 env: SawyerEnv,
                 info_callback=ppo_info,
                 use_max_path_len=True):
        self.env = env
        self._info_callback = info_callback
        self._use_max_path_len = use_max_path_len

    def step(self, action):
        goal_env_obs, r, done, info = self.env.step(action=action)
        return goal_env_obs.get('observation'), r, done, self._info_callback(
            info)

    def reset(self):
        goal_env_obs = self.env.reset()
        return goal_env_obs.get('observation')

    def render(self, mode='human'):
        self.env.render(mode)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space
