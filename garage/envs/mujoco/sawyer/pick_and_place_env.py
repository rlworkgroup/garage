from gym.envs.robotics import rotations
from gym.envs.robotics.utils import ctrl_set_action, mocap_set_action
from gym.spaces import Box
import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco import MujocoEnv
from garage.misc.overrides import overrides


class PickAndPlaceEnv(MujocoEnv, Serializable):

    FILE = 'pick_and_place.xml'

    def __init__(self,
                 initial_goal=None,
                 initial_qpos=None,
                 distance_threshold=0.05,
                 target_range=0.15,
                 sparse_reward=False,
                 control_method='position_control',
                 *args,
                 **kwargs):
        Serializable.__init__(self, *args, **kwargs)
        if initial_goal is None:
            self._initial_goal = np.array([0.0, 0.05, -0.1])
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
        super(PickAndPlaceEnv, self).__init__(*args, **kwargs)
        self.env_setup(self._initial_qpos)

        site_id = self.sim.model.site_name2id('target_pos')
        self.sim.model.site_pos[site_id] = self._initial_goal
        self.sim.forward()

    @overrides
    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self._control_method == 'torque_control':
            self.forward_dynamics(action)
        elif self._control_method == 'position_control':
            assert action.shape == (4, )
            action = action.copy()
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            pos_ctrl *= 0.5  # limit the action
            rot_ctrl = np.array([0., 1., 1., 0.])
            gripper_ctrl = -50 if gripper_ctrl < 0 else 10
            gripper_ctrl = np.array([gripper_ctrl, -gripper_ctrl])
            action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
            ctrl_set_action(self.sim, action)  # For gripper
            mocap_set_action(self.sim,
                             action)  # For pos control of the end effector
            self.sim.step()

        obs = self.get_current_obs()
        next_obs = obs['observation']
        achieved_goal = obs['achieved_goal']
        goal = obs['desired_goal']
        gripper_pos = obs['gripper_pos']
        reward = self._compute_reward(achieved_goal, goal, gripper_pos)
        done = (self._goal_distance(achieved_goal, goal) <
                self._distance_threshold)

        return Step(next_obs, reward, done)

    def _compute_reward(self, achieved_goal, goal, gripper_pos):
        # Compute distance between goal and the achieved goal.
        grasped = self._grasp()
        reward = 0
        if not grasped:
            # first phase: move towards the object
            d = self._goal_distance(gripper_pos, achieved_goal)
        else:
            d = self._goal_distance(achieved_goal, goal)
            if not self._grasped:
                reward += 400
                self._grasped = not self._grasped

        if self._sparse_reward:
            reward += -(d > self._distance_threshold).astype(np.float32)
        else:
            reward += -d

        if d < self._distance_threshold:
            reward += 4200
        return reward

    def _grasp(self):
        contacts = tuple()
        for coni in range(self.sim.data.ncon):
            con = self.sim.data.contact[coni]
            contacts += ((con.geom1, con.geom2), )
        if ((38, 2) in contacts or
            (2, 38) in contacts) and ((33, 2) in contacts or
                                      (38, 2) in contacts):
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
        if self._control_method == 'position_control':
            obs = np.concatenate([
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
            ])
        elif self._control_method == 'torque_control':
            obs = np.concatenate([
                object_pos.ravel(),
                object_rel_pos.ravel(),
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                qpos,
                qvel,
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
            return super(PickAndPlaceEnv, self).action_space()
        elif self._control_method == 'position_control':
            return Box(
                np.array([-0.1, -0.1, -0.1, -100]),
                np.array([0.1, 0.1, 0.1, 100]),
                dtype=np.float32)

    @overrides
    def reset(self, init_state=None):
        self._grasped = False
        return super(PickAndPlaceEnv, self).reset(init_state)['observation']

    def log_diagnostics(self, paths):
        pass
