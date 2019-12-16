import numpy as np
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_

class HalfCheetahVelEnv(HalfCheetahEnv_):
    def __init__(self):
        self.goal_velocity = 0
        super(HalfCheetahVelEnv, self).__init__()

    def sample_tasks(self, n_tasks):
        """Sample goal velocities.

        Returns:
            numpy.ndarray: An array of goal velocities.
        """
        return np.random.uniform(0.0, 2.0, (n_tasks, ))

    def set_task(self, task):
        """Set task.

        Args:
            task: task of the meta-learning environment
        """
        self.goal_velocity = task

    def get_task(self):
        """Get Task.

        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_velocity

    def step(self, action):
        """Run one timestep of the environment's dynamics."""
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.5 * 0.1 * np.square(action).sum()
        forward_vel = (xposafter - xposbefore) / self.dt
        reward_run = - np.abs(forward_vel - self.goal_velocity)
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(forward_vel=forward_vel, reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        """Get observations.

        Returns:
            numpy.ndarray: Observations.

        """
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
