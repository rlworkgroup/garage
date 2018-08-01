from collections import namedtuple
import os.path as osp

import gym
from gym.envs.robotics import rotations
from gym.envs.robotics.utils import reset_mocap2body_xpos, reset_mocap_welds
from gym.spaces import Box
import numpy as np

from garage.envs.mujoco import MujocoEnv
from garage.envs.mujoco.mujoco_env import MODEL_DIR
from garage.misc.overrides import overrides

Configuration = namedtuple(
    "Configuration",
    ["gripper_pos", "gripper_state", "object_grasped", "object_pos"])


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
                 success_fn=default_success_fn,
                 achieved_goal_fn=default_achieved_goal_fn,
                 desired_goal_fn=default_desired_goal_fn,
                 max_episode_steps=50,
                 completion_bonus=10,
                 distance_threshold=0.05,
                 for_her=False,
                 reward_type='dense',
                 control_method='task_space_control',
                 file_path='pick_and_place.xml',
                 *args,
                 **kwargs):
        """
        Sawyer Environment.  #noqa: E501
        :param args:
        :param kwargs:
        """

        self._start_goal_config = start_goal_config
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
        file_path = osp.join(MODEL_DIR, file_path)

        MujocoEnv.__init__(self, file_path=file_path, *args, **kwargs)
        self._env_setup()

    def _sample_start_goal(self):
        self._start_configuration, self._goal_configuration = self._start_goal_config(  # noqa: E501
        )

    def _env_setup(self):
        reset_mocap_welds(self.sim)
        self.sim.forward()

    def set_gripper_position(self, position):
        reset_mocap2body_xpos(self.sim)
        self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
        self.sim.data.set_mocap_pos('mocap', position)
        for _ in range(100):
            self.sim.step()
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
            self.sim.data.set_mocap_pos('mocap', position)
        self.sim.forward()

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
        else:
            return Box(
                np.array([-0.15, -0.15, -0.15, -1.]),
                np.array([0.15, 0.15, 0.15, 1.]),
                dtype=np.float32)

    @overrides
    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.get_obs()['observation'].shape,
            dtype=np.float32)

    def step(self, action):
        if self._control_method == "torque_control":
            self.forward_dynamics(action)
            self.sim.forward()
        else:
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_pos[
                0, :3] = self.sim.data.mocap_pos[0, :3] + action[:3]
            self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
            self.set_gripper_state(action[3])
            for _ in range(1):
                self.sim.step()
            self.sim.forward()

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
        if self._is_success:
            r = self._completion_bonus
            done = True

        info["r"] = r
        info["d"] = done

        return obs, r, done, info

    def set_gripper_state(self, state):
        # 1 = open, -1 = closed
        self.gripper_state = state
        state = (state + 1.) / 2.
        self.sim.data.ctrl[:] = np.array([state * 0.020833, -state * 0.020833])
        for _ in range(3):
            self.sim.step()
        self.sim.forward()

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

    @overrides
    def reset(self):
        self._step = 0
        super(SawyerEnv, self).reset()

        self._sample_start_goal()

        if self._start_configuration.object_grasped:
            self.set_gripper_state(1)  # open
            self.set_gripper_position(self._start_configuration.gripper_pos)
            self.set_object_position(self._start_configuration.gripper_pos)
            self.set_gripper_state(-1)  # close
        else:
            self.set_gripper_state(self._start_configuration.gripper_state)
            self.set_gripper_position(self._start_configuration.gripper_pos)
            self.set_object_position(self._start_configuration.object_pos)

        for _ in range(20):
            self.sim.step()

        return self.get_obs()

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self._reward_type == 'sparse':
            return (d < self._distance_threshold).astype(np.float32)
        return -d


def ppo_info(info):
    info["task"] = [1]
    ppo_infos = {"episode": info}
    return ppo_infos


class SawyerEnvWrapper():
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
