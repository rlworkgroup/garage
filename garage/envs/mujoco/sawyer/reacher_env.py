"""Reacher environment for the sawyer robot."""

from gym.envs.robotics.utils import reset_mocap2body_xpos
from gym.spaces import Box
import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco import MujocoEnv
from garage.misc.overrides import overrides


class ReacherEnv(MujocoEnv, Serializable):
    """Reacher Environment."""

    FILE = "reacher.xml"

    def __init__(self,
                 initial_goal=None,
                 initial_qpos=None,
                 distance_threshold=0.08,
                 target_range=0.15,
                 sparse_reward=False,
                 control_method='position_control',
                 *args,
                 **kwargs):
        """
        Reacher Environment.

        :param initial_goal: initial position to reach.
        :param initial_qpos: initial qpos for each joint.
        :param distance_threshold: distance threhold to define reached.
        :param target_range: delta range the goal is randomized
        :param sparse_reward: whether using sparse reward
        :param args
        :param kwargs
        """
        Serializable.quick_init(self, locals())
        if initial_goal is None:
            self._initial_goal = np.array([0.8, 0.0, 0.15])
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
        self._accumulated_reward = 0
        super(ReacherEnv, self).__init__(*args, **kwargs)
        self.env_setup(self._initial_qpos)

    @overrides
    def step(self, action):
        """
        Perform one step with action.

        :param action: the action to be performed, (x, y, z) only for pos_ctrl
        :return: next_obs, reward, done, info
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self._control_method == 'torque_control':
            self.forward_dynamics(action)
        elif self._control_method == 'position_control':
            assert action.shape == (3, )
            action = action.copy()
            action *= 0.1  # limit the action
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_pos[:] = self.sim.data.mocap_pos + action
            self.sim.step()
        else:
            raise NotImplementedError

        obs = self.get_current_obs()
        next_obs = obs['observation']
        achieved_goal = obs['achieved_goal']
        goal = obs['desired_goal']
        reward = self._compute_reward(achieved_goal, goal)
        done = (self._goal_distance(achieved_goal, goal) <
                self._distance_threshold)
        return Step(next_obs, reward, done)

    def _reset_target_visualization(self):
        """Reset the target visualization."""
        site_id = self.sim.model.site_name2id('target_pos')
        self.sim.model.site_pos[site_id] = self._initial_goal
        self.sim.forward()

    @overrides
    def reset(self, init_state=None):
        """Reset the environment."""
        self._accumulated_reward = 0
        self._reset_target_visualization()
        return super(ReacherEnv, self).reset(init_state)['observation']

    def _compute_reward(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = self._goal_distance(achieved_goal, goal)
        if self._sparse_reward:
            reward = -(d > self._distance_threshold).astype(np.float32)
        else:
            reward = -d

        if d < self._distance_threshold:
            reward += 500  # Completion
        return reward

    @overrides
    def get_current_obs(self):
        """
        Get the current observation.

        :return: current observation.
        """
        grip_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        achieved_goal = np.squeeze(grip_pos.copy())

        if self._control_method == 'position_control':
            obs = np.concatenate([
                grip_pos,
                grip_velp,
            ])
        elif self._control_method == 'torque_control':
            obs = np.concatenate([
                qpos,
                qvel,
            ])
        else:
            raise NotImplementedError

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self._goal
        }

    @staticmethod
    def _goal_distance(goal_a, goal_b):
        """
        Calculate the distance between two goals.

        :param goal_a: first goal.
        :param goal_b: second goal.
        :return: distance between goal a and b.
        """
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def sample_goal(self):
        """
        Sample goals.

        :return: the new sampled goal.
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
        Return a Space object.

        :return: observation space
        """
        return Box(
            -np.inf,
            np.inf,
            shape=self.get_current_obs()['observation'].shape,
            dtype=np.float32)

    @overrides
    @property
    def action_space(self):
        """Return an Action space."""
        if self._control_method == 'torque_control':
            return super(ReacherEnv, self).action_space()
        elif self._control_method == 'position_control':
            return Box(-1., 1., shape=(3, ), dtype=np.float32)
        else:
            raise NotImplementedError

    @overrides
    def close(self):
        """Make sure the environment start another viewer next time."""
        if self.viewer is not None:
            self.viewer = None

    def log_diagnostics(self, paths):
        """TODO: Logging."""
        pass
