from gym.spaces import Box
import numpy as np

from garage.envs.mujoco.sawyer.sawyer_env import Configuration, SawyerEnv
from garage.misc.overrides import overrides


class ReacherEnv(SawyerEnv):
    def __init__(self, goal_position, start_position=None, **kwargs):
        def generate_start_goal():
            nonlocal start_position
            if start_position is None:
                center = self.sim.data.get_geom_xpos('target2')
                start_position = np.concatenate([center[:2], [0.15]])

            start = Configuration(
                gripper_pos=start_position,
                gripper_state=1,
                object_grasped=False,
                object_pos=[0, 0, -1])
            goal = Configuration(
                gripper_pos=goal_position,
                gripper_state=1,
                object_grasped=False,
                object_pos=[0, 0, -1])

            return start, goal

        super(ReacherEnv, self).__init__(
            start_goal_config=generate_start_goal, **kwargs)

    def get_obs(self):
        gripper_pos = self.gripper_position
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        object_pos = self.object_position
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velp -= grip_velp
        grasped = self.has_object
        obs = np.concatenate([gripper_pos])

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
        return Box(
            np.array([-0.5, -0.5, -0.5, -1.]),
            np.array([0.5, 0.5, 0.5, 1.]),
            dtype=np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info: dict):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self._reward_type == 'sparse':
            return (d < self._distance_threshold).astype(np.float32)

        return .01 - d
