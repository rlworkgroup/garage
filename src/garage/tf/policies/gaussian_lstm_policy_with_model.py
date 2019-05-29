"""GaussianLSTMPolicy with GaussianLSTMModel."""
import akro.tf
import numpy as np
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.models import GaussianLSTMModel
from garage.tf.policies.base2 import StochasticPolicy2


class GaussianLSTMPolicyWithModel(StochasticPolicy2):
    """
    GaussianLSTMPolicy with GaussianLSTMModel.

    A policy that contains a LSTM to make prediction based on
    a gaussian distribution.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Model name, also the variable scope.
        hidden_dim (int): Hidden dimension for LSTM cell for mean.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        recurrent_nonlinearity (callable): Activation function for recurrent
            layers. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        recurrent_w_init (callable): Initializer function for the weight
            of recurrent layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        hidden_state_init (callable): Initializer function for the
            initial hidden state. The functino should return a tf.Tensor.
        hidden_state_init_trainable (bool): Bool for whether the initial
            hidden state is trainable.
        cell_state_init (callable): Initializer function for the
            initial cell state. The functino should return a tf.Tensor.
        cell_state_init_trainable (bool): Bool for whether the initial
            cell state is trainable.
        forget_bias (bool): If True, add 1 to the bias of the forget gate at
            initialization. It's used to reduce the scale of forgetting at the
            beginning of the training.
        learn_std (bool): Is std trainable.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        init_std (float): Initial value for std.
        layer_normalization (bool): Bool for using layer normalization or not.
        state_include_action (bool): Whether the state includes action.
            If True, input dimension will be
            (observation dimension + action dimension).
    """

    def __init__(self,
                 env_spec,
                 hidden_dim=32,
                 name='GaussianLSTMPolicyWithModel',
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 recurrent_nonlinearity=tf.nn.sigmoid,
                 recurrent_w_init=tf.glorot_uniform_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 hidden_state_init=tf.zeros_initializer(),
                 hidden_state_init_trainable=False,
                 cell_state_init=tf.zeros_initializer(),
                 cell_state_init_trainable=False,
                 forget_bias=True,
                 learn_std=True,
                 std_share_network=False,
                 init_std=1.0,
                 layer_normalization=False,
                 state_include_action=True):
        if not isinstance(env_spec.action_space, akro.tf.Box):
            raise ValueError('GaussianLSTMPolicy only works with '
                             'akro.tf.Box action space, but not {}'.format(
                                 env_spec.action_space))
        super().__init__(name, env_spec)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._hidden_dim = hidden_dim
        self._state_include_action = state_include_action

        if state_include_action:
            self._input_dim = self._obs_dim + self._action_dim
        else:
            self._input_dim = self._obs_dim

        self.model = GaussianLSTMModel(
            output_dim=self._action_dim,
            hidden_dim=hidden_dim,
            name='GaussianLSTMModel',
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            recurrent_nonlinearity=recurrent_nonlinearity,
            recurrent_w_init=recurrent_w_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            hidden_state_init=hidden_state_init,
            hidden_state_init_trainable=hidden_state_init_trainable,
            cell_state_init=cell_state_init,
            cell_state_init_trainable=cell_state_init_trainable,
            forget_bias=forget_bias,
            layer_normalization=layer_normalization,
            learn_std=learn_std,
            std_share_network=std_share_network,
            init_std=init_std)

        self._initialize()

    def _initialize(self):
        obs_ph = tf.placeholder(
            tf.float32, shape=(None, None, self._input_dim))
        step_input_var = tf.placeholder(
            shape=(None, self._input_dim), name='step_input', dtype=tf.float32)
        step_hidden_var = tf.placeholder(
            shape=(None, self._hidden_dim),
            name='step_hidden_input',
            dtype=tf.float32)
        step_cell_var = tf.placeholder(
            shape=(None, self._hidden_dim),
            name='step_cell_input',
            dtype=tf.float32)

        with tf.variable_scope(self.name) as vs:
            self._variable_scope = vs
            self.model.build(obs_ph, step_input_var, step_hidden_var,
                             step_cell_var)

        self._f_step_mean_std = tf.get_default_session().make_callable(
            [
                self.model.networks['default'].sample,
                self.model.networks['default'].step_mean,
                self.model.networks['default'].step_log_std,
                self.model.networks['default'].step_hidden,
                self.model.networks['default'].step_cell
            ],
            feed_list=[step_input_var, step_hidden_var, step_cell_var])

        self.prev_actions = None
        self.prev_hiddens = None
        self.prev_cells = None

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    def dist_info_sym(self, obs_var, state_info_vars, name=None):
        """
        Symbolic graph of the distribution.

        Args:
            obs_var (tf.Tensor): Tensor input for symbolic graph.
            state_info_vars (dict): Extra state information, e.g.
                previous action.
            name (str): Name for symbolic graph.

        """
        if self._state_include_action:
            prev_action_var = state_info_vars['prev_action']
            prev_action_var = tf.cast(prev_action_var, tf.float32)
            all_input_var = tf.concat(
                axis=2, values=[obs_var, prev_action_var])
        else:
            all_input_var = obs_var

        with tf.variable_scope(self._variable_scope):
            _, mean_var, _, log_std_var, _, _, _, _, _, _ = self.model.build(
                all_input_var,
                self.model.networks['default'].step_input,
                self.model.networks['default'].step_hidden_input,
                self.model.networks['default'].step_cell_input,
                name=name)

        return dict(mean=mean_var, log_std=log_std_var)

    def reset(self, dones=None):
        """
        Reset the policy.

        Args:
            dones (numpy.ndarray): Bool that indicates terminal state(s).

        """
        if not dones:
            dones = np.array([True])
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones),
                                          self.action_space.flat_dim))
            self.prev_hiddens = np.zeros((len(dones), self._hidden_dim))
            self.prev_cells = np.zeros((len(dones), self._hidden_dim))

        self.prev_actions[dones] = 0.
        self.prev_hiddens[dones] = self.model.networks[
            'default'].init_hidden.eval()
        self.prev_cells[dones] = self.model.networks['default'].init_cell.eval(
        )

    @overrides
    def get_action(self, observation):
        """
        Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            action (numpy.ndarray): Predicted action.
            agent_info (dict[numpy.ndarray]): Mean and log std of the
                distribution obtained after observing the given observation.

        """
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @overrides
    def get_actions(self, observations):
        """
        Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            actions (numpy.ndarray): Predicted actions.
            agent_infos (dict[numpy.ndarray]): Mean and log std of the
                distributions obtained after observing the given observations.

        """
        flat_obs = self.observation_space.flatten_n(observations)
        if self._state_include_action:
            assert self.prev_actions is not None
            all_input = np.concatenate([flat_obs, self.prev_actions], axis=-1)
        else:
            all_input = flat_obs
        samples, means, log_stds, hidden_vec, cell_vec = self._f_step_mean_std(
            all_input, self.prev_hiddens, self.prev_cells)
        samples = self.action_space.unflatten_n(samples)
        prev_actions = self.prev_actions
        self.prev_actions = samples
        self.prev_hiddens = hidden_vec
        self.prev_cells = cell_vec
        agent_info = dict(mean=means, log_std=log_stds)
        if self._state_include_action:
            agent_info['prev_action'] = np.copy(prev_actions)
        return samples, agent_info

    @property
    def recurrent(self):
        """Recurrent or not."""
        return True

    @property
    def distribution(self):
        """Policy distribution."""
        return self.model.networks['default'].dist

    @property
    def state_info_specs(self):
        """State info specification."""
        if self._state_include_action:
            return [
                ('prev_action', (self._action_dim, )),
            ]
        else:
            return []

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['_f_step_mean_std']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        super().__setstate__(state)
        self._initialize()
