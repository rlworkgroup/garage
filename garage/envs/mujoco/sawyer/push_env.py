import numpy as np

from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnv, Configuration


class PushEnv(SawyerEnv):
    def __init__(self, direction="up", **kwargs):
        def start_goal_config():
            # center = self.sim.data.get_geom_xpos('target2')
            xy = np.random.uniform([0.3, 0.6], [-0.2, 0.2], 2)
            start = Configuration(
                gripper_pos=np.concatenate([xy, [0.35]]),
                gripper_state=0,
                object_grasped=False,
                object_pos=np.concatenate([xy, [0.03]]))
            d = 0.2
            delta = np.array({
                "up":    ( d,  0),
                "down":  (-d,  0),
                "left":  ( 0, -d),
                "right": ( 0,  d)
            }[direction])
            goal = Configuration(
                gripper_pos=np.concatenate([xy + delta, [0.35]]),
                gripper_state=0,
                object_grasped=False,
                object_pos=np.concatenate([xy + delta, [0.03]]))
            return start, goal

        def achieved_goal_fn(env: SawyerEnv):
            return env.object_position

        def desired_goal_fn(env: SawyerEnv):
            return env._goal_configuration.object_pos

        super(PushEnv, self).__init__(
            start_goal_config=start_goal_config,
            achieved_goal_fn=achieved_goal_fn,
            desired_goal_fn=desired_goal_fn,
            file_path="push.xml",
            **kwargs)

    def get_obs(self):
        gripper_pos = self.gripper_position
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        object_pos = self.object_position
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velp -= grip_velp
        grasped = self.has_object
        obs = np.concatenate([gripper_pos, object_pos])

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
