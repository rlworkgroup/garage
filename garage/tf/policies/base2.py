"""Policy base classes without Parameterized."""

from garage.misc.tensor_utils import flatten_tensors, unflatten_tensors


class Policy2:
    """
    Policy base class without Parameterzied.

    Args:
        name (str): Policy name, also the variable scope.
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    """

    def __init__(self, name, env_spec):
        self._name = name
        self._env_spec = env_spec
        self._variable_scope = None

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

    def get_params(self, trainable=True):
        """Get the trainable variables."""
        return self.get_trainable_vars()

    def get_param_dtypes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [
                val.dtype for val in param_values
            ]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [
                val.shape for val in param_values
            ]
        return self._cached_param_shapes[tag_tuple]

    def get_param_values(self, **tags):
        params = self.get_params(**tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def set_param_values(self, param_values, name=None, **tags):
        ops = []
        feed_dict = dict()
        param_values = unflatten_tensors(param_values,
                                         self.get_param_shapes(**tags))
        for param, value in zip(self.get_params(**tags), param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(
                    dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]] = value

        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def flat_to_params(self, flattened_params, **tags):
        return unflatten_tensors(flattened_params,
                                 self.get_param_shapes(**tags))


class StochasticPolicy2(Policy2):
    """StochasticPolicy."""

    @property
    def distribution(self):
        """Distribution."""
        raise NotImplementedError

    def dist_info_sym(self, obs_var, state_info_vars, name='dist_info_sym'):
        """
        Symbolic graph of the distribution.

        Return the symbolic distribution information about the actions.
        Args:
            obs_var (tf.Tensor): symbolic variable for observations
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation.
            name (str): Name of the symbolic graph.

        :return:
        """
        raise NotImplementedError

    def dist_info(self, obs, state_infos):
        """
        Distribution info.

        Return the distribution information about the actions.

        Args:
            obs_var (tf.Tensor): observation values
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation
        """
        raise NotImplementedError
