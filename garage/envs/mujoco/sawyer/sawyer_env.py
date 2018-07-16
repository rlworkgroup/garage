"""An interface for all sawyer robot environment."""

import gym
from gym.envs.robotics.utils import reset_mocap_welds

from garage.envs.mujoco import MujocoEnv
from garage.misc.overrides import overrides


class SawyerEnv(MujocoEnv, gym.GoalEnv):
    """Sawyer Robot Environments."""

    def __init__(self,
                 initial_goal=None,
                 initial_qpos=None,
                 target_range=0.15,
                 *args,
                 **kwargs):
        """
        Sawyer Environment.

        :param initial_goal: The initial goal for the goal environment.
        :param initial_qpos: The initial position for each joint.
        :param target_range: delta range the goal is randomized.
        :param args:
        :param kwargs:
        """
        self._initial_goal = initial_goal
        self._initial_qpos = initial_qpos
        self._target_range = target_range
        self._goal = self._initial_goal
        MujocoEnv.__init__(self, *args, **kwargs)
        if initial_qpos is not None:
            self.env_setup(initial_qpos)

    @overrides
    def close(self):
        """Make sure the environment start another viewer next time."""
        if self.viewer is not None:
            self.viewer = None

    def log_diagnostics(self, paths):
        """TODO: Logging."""
        pass

    def env_setup(self, initial_qpos):
        """Set up the robot with initial qpos."""
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        reset_mocap_welds(self.sim)
        self.sim.forward()
