"""
robot Interface
"""


class Robot:
    def reset(self):
        """
        User uses this to reset the robot at the beginning of every training
        episode.
        """
        raise NotImplementedError

    def send_command(self, commands):
        """
        User uses this to send commands to robot.
        :param commands: np.array(action_space.shape)
                    based on action_space
        """
        raise NotImplementedError

    @property
    def action_space(self):
        """
        User uses this to get robot's action_space.
        """
        raise NotImplementedError

    def get_observation(self):
        """
        User uses this to get the most recent observation of robot.
        :return robot_observation: np.array([...])
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """
        User uses this to get robot's observation_space
        """
        raise NotImplementedError
