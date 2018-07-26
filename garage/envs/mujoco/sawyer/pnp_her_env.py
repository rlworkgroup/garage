from gym.envs.robotics import rotations
from gym.envs.robotics.utils import ctrl_set_action, mocap_set_action
from gym.spaces import Box
import mujoco_py
import numpy as np

from garage.core import Serializable
from garage.envs.mujoco import MujocoEnv
from garage.misc.overrides import overrides


class PnpHerEnv(MujocoEnv, Serializable):

    FILE = 'pick_and_place_eric.xml'

    def __init__(self,
                 initial_goal=None,
                 initial_qpos=None,
                 distance_threshold=0.05,
                 target_range=0.15,
                 sparse_reward=True,
                 control_method='position_control',
                 *args,
                 **kwargs):
        Serializable.__init__(self, *args, **kwargs)
        if initial_goal is None:
            self._initial_goal = np.array([0.8, 0.0, 0.])
        else:
            self._initial_goal = initial_goal
        if initial_qpos is not None:
            self._initial_qpos = initial_qpos
        else:
            self._initial_qpos = {
                'right_j0': -0.140923828125,
                'right_j1': -1.2789248046875,
                'right_j2': -3.043166015625,
                'right_j3': -2.139623046875,
                'right_j4': -0.047607421875,
                'right_j5': -0.7052822265625,
                'right_j6': -1.4102060546875,
            }
        self._distance_threshold = distance_threshold
        self._target_range = target_range
        self._sparse_reward = sparse_reward
        self._control_method = control_method
        self._goal = self._initial_goal
        self._grasped = False
        super(PnpHerEnv, self).__init__(*args, **kwargs)
        self.env_setup(self._initial_qpos)
        self._mj_viewer = mujoco_py.MjViewer(self.sim)
        self.len = 0
        self.rew = 0
        self._grasped_time = 0
        self._max_episode_steps = 400

    @overrides
    def step(self, action: np.ndarray, collision_penalty=0):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self._control_method == 'torque_control':
            self.forward_dynamics(action)
        elif self._control_method == 'position_control':
            assert action.shape == (4, )
            action = action.copy()
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            pos_ctrl *= 0.1  # limit the action
            rot_ctrl = np.array([0., 1., 1., 0.])
            gripper_ctrl = -0.05 if gripper_ctrl <= 0 else 0.05
            gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
            action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
            ctrl_set_action(self.sim, action)  # For gripper
            mocap_set_action(self.sim,
                             action)  # For pos control of the end effector
            self.sim.step()

        grasped = self._grasp()
        if grasped:
            self._grasped_time += 1
        self._grasped = grasped

        obs = self.get_current_obs()
        # next_obs = obs['observation']
        next_obs = obs
        achieved_goal = obs['achieved_goal']
        goal = obs['desired_goal']
        gripper_pos = obs['gripper_pos']
        reward = self._compute_reward(self._grasped, self._grasped_time,
                                      achieved_goal, goal, gripper_pos)
        collided = self._is_collided()
        if collided:
            reward += collision_penalty

        # done = (self._goal_distance(achieved_goal, goal) <
        #         self._distance_threshold) or (collided and collision_penalty < 0)
        done = (self._goal_distance(achieved_goal, goal) <
                self._distance_threshold)
        self.rew += reward
        self.len += 1
        info = dict()
        if done or self.len >= self._max_episode_steps:
            info = dict(episode=dict(r=self.rew, l=self.len))

            next_obs = self.reset()
        info['is_success'] = done
        return next_obs, reward, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.

        # if not grasped:
        # first phase: move towards the object
        # d = self._goal_distance(gripper_pos,
        #                         achieved_goal) + self._goal_distance(
        #                             gripper_pos[:2], achieved_goal[:2])
        # if not self._grasped and d < 0.12 and d > 0.10:
        #         reward += 0.5
        # else:
        d = self._goal_distance(achieved_goal, desired_goal)
        reward = np.zeros_like(d)
        # reward += 10
        # if grasp_time == 1:
        #     reward += 600
        #     print('Already grasped!')

        if self._sparse_reward:
            reward = -(d > self._distance_threshold).astype(np.float32)
        # if grasped and \
        # if d < self._distance_threshold:
        #     reward += 3000
        return reward

    def _compute_reward(self, grasped, grasp_time, achieved_goal, goal,
                        gripper_pos):
        # Compute distance between goal and the achieved goal.
        reward = 0
        # if not grasped:
        # first phase: move towards the object
        # d = self._goal_distance(gripper_pos,
        #                         achieved_goal) + self._goal_distance(
        #                             gripper_pos[:2], achieved_goal[:2])
        # if not self._grasped and d < 0.12 and d > 0.10:
        #         reward += 0.5
        # else:
        d = self._goal_distance(achieved_goal, goal)
        # reward += 10
        # if grasp_time == 1:
        #     reward += 600
        #     print('Already grasped!')

        if self._sparse_reward:
            reward += -(d > self._distance_threshold).astype(np.float32)
        else:
            reward += -d
        # if grasped and \
        # if d < self._distance_threshold:
        #     reward += 3000
        return reward

    def _grasp(self):
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
                (finger_id_2, object_id) in contacts):
            return True
        else:
            return False

    def sample_goal(self):
        """
        Sample goals
        :return: the new sampled goal
        """
        goal = self._initial_goal.copy()

        random_goal_delta = np.random.uniform(
            -self._target_range, self._target_range, size=2)
        goal[:2] += random_goal_delta
        self._goal = goal
        return goal

    @overrides
    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        return Box(
            -np.inf,
            np.inf,
            shape=self.get_current_obs()['observation'].shape,
            dtype=np.float32)

    @overrides
    def get_current_obs(self):
        grip_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        object_pos = self.sim.data.get_geom_xpos('object0')
        object_rot = rotations.mat2euler(
            self.sim.data.get_geom_xmat('object0'))
        object_velp = self.sim.data.get_geom_xvelp('object0') * dt
        object_velr = self.sim.data.get_geom_xvelr('object0') * dt
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp

        achieved_goal = np.squeeze(object_pos.copy())

        grasp = np.array([int(self._grasped)])
        if self._control_method == 'position_control':
            obs = np.concatenate([
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(), grip_velp, grasp
            ])
        elif self._control_method == 'torque_control':
            obs = np.concatenate([
                object_pos.ravel(),
                object_rel_pos.ravel(),
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(), qpos, qvel, grasp
            ])
        else:
            raise NotImplementedError

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self._goal,
            'gripper_pos': grip_pos,
        }

    @staticmethod
    def _goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    @property
    def action_space(self):
        if self._control_method == 'torque_control':
            action_space = super(PnpHerEnv, self).action_space()
        elif self._control_method == 'position_control':
            action_space = Box(
                np.array([-1, -1, -1, -100]),
                np.array([1, 1, 1, 100]),
                dtype=np.float32)
        return action_space

    def _reset_target_visualization(self):
        site_id = self.sim.model.site_name2id('target_pos')
        self.sim.model.site_pos[site_id] = self._goal
        self.sim.forward()

    @overrides
    def reset(self, init_state=None):
        self._grasped = False
        self.len = 0
        self.rew = 0
        self._reset_target_visualization()
        self._grasped_time = 0
        return super(PnpHerEnv, self).reset(init_state)

    def log_diagnostics(self, paths):
        """TODO: Logging."""
        pass

    def _is_collided(self):
        """Detect collision"""
        d = self.sim.data
        table_id = self.sim.model.geom_name2id('table')
        object_id = self.sim.model.geom_name2id('object0')

        for i in range(d.ncon):
            con = d.contact[i]
            if table_id == con.geom1 and con.geom2 != object_id:
                return True
            if table_id == con.geom2 and con.geom1 != object_id:
                return True
        return False
