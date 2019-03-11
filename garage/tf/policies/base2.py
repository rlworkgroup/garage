"""Policy base classes without Parameterized."""
import tensorflow as tf


class Policy2:
    """
    Policy base class without Parameterzied.

    Args:
        env_spec: Environment specification.

    """

    def __init__(self, name, env_spec):
        self._name = name
        self._env_spec = env_spec
        self._variable_scope = tf.VariableScope(name)

    # Should be implemented by all policies

    def get_action(self, observation):
        """Get action given observation."""
        raise NotImplementedError

    def get_actions(self, observations):
        """Get actions given observations."""
        raise NotImplementedError

    def reset(self, dones=None):
        """Reset policy."""
        pass

    @property
    def name(self):
        return self._name

    @property
    def vectorized(self):
        """
        Boolean for vectorized.

        Indicates whether the policy is vectorized. If True, it should
        implement get_actions(), and support resetting
        with multiple simultaneous states.
        """
        return False

    @property
    def observation_space(self):
        """Observation space."""
        return self._env_spec.observation_space

    @property
    def action_space(self):
        """Policy action space."""
        return self._env_spec.action_space

    @property
    def env_spec(self):
        """Policy environment specification."""
        return self._env_spec

    @property
    def recurrent(self):
        """Boolean indicating if the policy is recurrent."""
        return False

    def log_diagnostics(self, paths):
        """Log extra information per iteration based on the collected paths."""
        pass

    @property
    def state_info_keys(self):
        """
        State info keys.

        Return keys for the information related to the policy's state when
        taking an action.
        :return:
        """
        return [k for k, _ in self.state_info_specs]

    @property
    def state_info_specs(self):
        """
        State info specifcation.

        Return keys and shapes for the information related to the policy's
        state when taking an action.
        :return:
        """
        return list()

    def terminate(self):
        """Clean up operation."""
        pass

    def get_trainable_vars(self):
        """Get trainable vars."""
        return self._variable_scope.trainable_variables()

    def get_global_vars(self):
        """Get global vars."""
        return self._variable_scope.global_variables()

    def get_regularizable_vars(self):
        """Get regularizable vars."""
        reg_vars = [
            var for var in self.get_trainable_vars()
            if 'W' in var.name and 'output' not in var.name
        ]
        return reg_vars


class StochasticPolicy2(Policy2):
    """StochasticPolicy."""

    @property
    def distribution(self):
        """Distribution."""
        raise NotImplementedError

    def dist_info_sym(self, obs_var, state_info_vars, name="dist_info_sym"):
        """
        Symbolic graph of the distribution.

        Return the symbolic distribution information about the actions.
        :param obs_var: symbolic variable for observations
        :param state_info_vars: a dictionary whose values should contain
         information about the state of the policy at
        the time it received the observation
        :return:
        """
        raise NotImplementedError

    def dist_info(self, obs, state_infos):
        """
        Distribution info.

        Return the distribution information about the actions.
        :param obs_var: observation values
        :param state_info_vars: a dictionary whose values should contain
         information about the state of the policy at the time it received the
         observation
        :return:
        """
        raise NotImplementedError
