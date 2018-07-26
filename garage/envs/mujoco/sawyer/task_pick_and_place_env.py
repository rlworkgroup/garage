from garage.envs.base import Step
from garage.misc.overrides import overrides
from garage.envs.mujoco.sawyer import PnpHerEnv

import gym
from gym.spaces import Box
from gym.envs.robotics.utils import reset_mocap2body_xpos

import numpy as np

TASKS = [(0.6, -0.3, 0.003, True), (0.6, 0, 0.003, True),
         (0.6, 0.3, 0.003, True), (0.6, -0.3, 0.15, False),
         (0.6, 0, 0.15, False), (0.6, 0.3, 0.15, False)]


class TaskPickAndPlaceEnv(PnpHerEnv):
    def __init__(self,
                 task=0,
                 control_method="position_control",
                 *args,
                 **kwargs):
        self._task = task
        self.onehot = np.zeros(len(TASKS))
        self.onehot[self._task] = 1
        self._step = 0
        self._start_pos = np.array([0.8, 0.0, 0.15])
        super().__init__(control_method=control_method, *args, **kwargs)
        self._distance_threshold = 0.03
        reset_mocap2body_xpos(self.sim)

        self.env_setup(self._initial_qpos)

        self.init_qpos = self.sim.data.qpos
        self.init_qvel = self.sim.data.qvel
        self.init_qacc = self.sim.data.qacc
        self.init_ctrl = self.sim.data.ctrl

        self._goal = np.array(TASKS[task][:3])
        self.set_position(self._start_pos)
        print(
            "Instantiating TaskPickAndPlaceEnv (task = %i, control_mode = %s)"
            % (self._task, self._control_method))

    @overrides
    @property
    def action_space(self):
        if self._control_method == 'torque_control':
            return super(TaskPickAndPlaceEnv, self).action_space
        elif self._control_method == 'position_control':
            # specify lower action limits
            return Box(-0.03, 0.03, shape=(4, ), dtype=np.float32)
        else:
            raise NotImplementedError()

    @overrides
    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.get_obs().shape,
            dtype=np.float32)

    def get_obs(self):
        if self._control_method == 'torque_control':
            return np.concatenate((self.position, self.sim.data.qpos))
        else:
            return self.position

    @overrides
    def step(self, action, collision_penalty=0):
        # no clipping / rescaling of actions
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        # rot_ctrl = np.array([1., 0., 1., 0.])
        # action = np.concatenate([action, rot_ctrl])
        # action, _ = np.split(action, (self.sim.model.nmocap * 7,))
        # action = action.reshape(self.sim.model.nmocap, 7)

        # pos_delta = action[:, :3]
        if self._control_method == "torque_control":
            self.forward_dynamics(action)
            self.sim.forward()
        else:
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_pos[:] = self.sim.data.mocap_pos + action[:3]
            for _ in range(50):
                self.sim.step()
            self._step += 1

        # obs = self.get_current_obs()
        # achieved_goal = obs['achieved_goal']
        # goal = obs['desired_goal']
        achieved_goal = self.position
        # reward = self._compute_reward(achieved_goal, goal)

        obs = self.get_obs()

        achieved_dist = self._goal_distance(achieved_goal, self._goal)
        # reward = rewards._sigmoids(self._goal_distance(achieved_goal, goal) / self._goal_distance(self.initial_pos, goal), 0., "cosine")
        # reward = 1. - achieved_dist / self._goal_distance(self._start_pos, self._goal) / 2.  # before
        # reward = 1. - achieved_dist / self._goal_distance(np.zeros(3), self._goal) / 2.
        # TODO sparse reward
        reward = 1. - achieved_dist / self._goal_distance(
            self._start_pos, self._goal)

        # print(self.initial_pos, achieved_goal)

        done = (achieved_dist < self._distance_threshold)

        if done:
            reward = 20.  # 20.

        info = {
            "episode": {
                "l": self._step,
                "r": reward,
                "d": done,
                "position": self.position,
                "task": np.copy(self.onehot)
            }
        }
        # just take gripper position as observation
        return Step(obs, reward, False, **info)

    @overrides
    def reset(self, init_state=None):
        self._step = 0
        super(TaskPickAndPlaceEnv, self).reset(init_state)[:3]
        self.set_position(self._start_pos)
        self.select_task(self._task)
        return self.get_obs()

    @property
    def task(self):
        return self._task

    @property
    def goal(self):
        return self._goal

    @property
    def position(self):
        return self.sim.data.get_site_xpos('grip')

    def set_position(self, pos):
        # self.sim.data.set_mocap_pos('mocap', pos[:3])
        # self.sim.data.set_site_pos('object0', pos)

        # self.sim.step()
        # for _ in range(200):
        #     self.sim.step()

        reset_mocap2body_xpos(self.sim)
        self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
        # gripper_ctrl = -50 if gripper_ctrl < 0 else 10
        gripper_ctrl = np.array([0, 0])
        # action = np.concatenate([pos, rot_ctrl, gripper_ctrl])
        # ctrl_set_action(self.sim, action)  # For gripper
        # mocap_set_action(self.sim, action)
        # reset_mocap2body_xpos(self.sim)
        self.sim.data.set_mocap_pos('mocap', pos)
        for _ in range(2000):
            self.sim.step()

            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
            self.sim.data.set_mocap_pos('mocap', pos)
            # self.sim.forward()
        # # self.sim.data.mocap_pos[:] = pos
        # reset_mocap2body_xpos(self.sim)
        # print("Set position to", pos)
        # print("SawyerReach Servo Error:", np.linalg.norm(pos-grip_pos))
        self.sim.forward()

    @property
    def start_position(self):
        return self._start_pos.copy()

    def set_start_position(self, point):
        self._start_pos[:] = np.copy(point)

    def select_next_task(self):
        self._task = (self._task + 1) % len(TASKS)
        self.onehot = np.zeros(len(TASKS))
        self.onehot[self._task] = 1
        self._goal = np.array(TASKS[self._task][:3], dtype=np.float32)
        # site_id = self.sim.model.site_name2id('target_pos')
        # self.sim.model.site_pos[site_id] = self._goal
        self.sim.forward()
        return self._task

    def set_gripper_state(self, state):
        state = np.clip(state, 0., 1.)
        # self.sim.data.set_joint_qpos('r_gripper_l_finger_joint', state * 0.020833)
        # self.sim.data.set_joint_qpos('r_gripper_r_finger_joint', -state * 0.020833)

        self.sim.data.ctrl[:] = np.array([state * 0.020833, -state * 0.020833])
        for _ in range(self.frame_skip):
            self.sim.step()
        # self.sim.forward()
        new_com = self.sim.data.subtree_com[0]
        self.dcom = new_com - self.current_com
        self.current_com = new_com
        # self.sim

    def select_task(self, task: int):
        self._task = task
        self.onehot = np.zeros(len(TASKS))
        self.onehot[self._task] = 1
        self._goal = np.array(TASKS[self._task][:3], dtype=np.float32)

        if TASKS[self._task][3]:
            # pick
            # object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            object_qpos = np.concatenate((TASKS[self._task][:3], [1, 0, 0, 0]))
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            r = np.random.randn(3) * 0.2
            # print(r)
            self.set_position(self._start_pos)  # + np.random.randn(3) * 0.2)
            self.set_gripper_state(0)  # closed
        else:
            # place
            pos = np.array(self._start_pos[:3])  # + np.random.randn(3) * 0.2
            self._goal = np.array(pos, dtype=np.float32)
            self.set_position(self._goal)
            self.set_gripper_state(1)  # open
            for _ in range(20):
                self.sim.step()
            # place object
            offset = np.array([0, 0, -.1])
            object_qpos = np.concatenate((pos + offset, [1, 0, 0, 0]))
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            for _ in range(20):
                self.sim.step()
            self.set_gripper_state(0)  # closed

        for _ in range(20):
            self.sim.step()
            # self.sim.forward()

        # site_id = self.sim.model.site_name2id('target_pos')
        # self.sim.model.site_pos[site_id] = self._goal
        self.sim.forward()
        return self._task
