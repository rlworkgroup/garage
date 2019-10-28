"""CategoricalLSTMPolicy with model."""
import akro
import numpy as np
import tensorflow as tf

from garage.tf.distributions import RecurrentCategorical
from garage.tf.models import LSTMModel
from garage.tf.policies import StochasticPolicy


class CategoricalLSTMPolicy(StochasticPolicy):
    """CategoricalLSTMPolicy

    A policy that contains a LSTM to make prediction based on
    a categorical distribution.

    It only works with akro.Discrete action space.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Policy name, also the variable scope.
        hidden_dim (int): Hidden dimension for LSTM cell.
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
        state_include_action (bool): Whether the state includes action.
            If True, input dimension will be
            (observation dimension + action dimension).
        forget_bias (bool): If True, add 1 to the bias of the forget gate
            at initialization. It's used to reduce the scale of forgetting at
            the beginning of the training.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 env_spec,
                 name='CategoricalLSTMPolicy',
                 hidden_dim=32,
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 recurrent_nonlinearity=tf.nn.sigmoid,
                 recurrent_w_init=tf.glorot_uniform_initializer(),
                 output_nonlinearity=tf.nn.softmax,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 hidden_state_init=tf.zeros_initializer(),
                 hidden_state_init_trainable=False,
                 cell_state_init=tf.zeros_initializer(),
                 cell_state_init_trainable=False,
                 state_include_action=True,
                 forget_bias=True,
                 layer_normalization=False):
        if not isinstance(env_spec.action_space, akro.Discrete):
            raise ValueError('CategoricalLSTMPolicy only works'
                             'with akro.Discrete action space.')

        super().__init__(name, env_spec)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.n
        self._hidden_dim = hidden_dim
        self._state_include_action = state_include_action
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._hidden_state_init = hidden_state_init
        self._cell_state_init = cell_state_init

        if state_include_action:
            self._input_dim = self._obs_dim + self._action_dim
        else:
            self._input_dim = self._obs_dim

        self.model = LSTMModel(
            output_dim=self._action_dim,
            hidden_dim=self._hidden_dim,
            name='prob_network',
            forget_bias=forget_bias,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            recurrent_nonlinearity=recurrent_nonlinearity,
            recurrent_w_init=recurrent_w_init,
            hidden_state_init=hidden_state_init,
            hidden_state_init_trainable=hidden_state_init_trainable,
            cell_state_init=cell_state_init,
            cell_state_init_trainable=cell_state_init_trainable,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

        self._prev_actions = None
        self._prev_hiddens = None
        self._prev_cells = None
        self._initialize()

    def _initialize(self):
        obs_ph = tf.compat.v1.placeholder(tf.float32,
                                          shape=(None, None, self._input_dim))
        step_input_var = tf.compat.v1.placeholder(shape=(None,
                                                         self._input_dim),
                                                  name='step_input',
                                                  dtype=tf.float32)
        step_hidden_var = tf.compat.v1.placeholder(shape=(None,
                                                          self._hidden_dim),
                                                   name='step_hidden_input',
                                                   dtype=tf.float32)
        step_cell_var = tf.compat.v1.placeholder(shape=(None,
                                                        self._hidden_dim),
                                                 name='step_cell_input',
                                                 dtype=tf.float32)

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            self.model.build(obs_ph, step_input_var, step_hidden_var,
                             step_cell_var)

        self._f_step_prob = tf.compat.v1.get_default_session().make_callable(
            [
                self.model.networks['default'].step_output,
                self.model.networks['default'].step_hidden,
                self.model.networks['default'].step_cell
            ],
            feed_list=[step_input_var, step_hidden_var, step_cell_var])

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    def dist_info_sym(self, obs_var, state_info_vars, name=None):
        """Symbolic graph of the distribution."""
        if self._state_include_action:
            prev_action_var = state_info_vars['prev_action']
            prev_action_var = tf.cast(prev_action_var, tf.float32)
            all_input_var = tf.concat(axis=2,
                                      values=[obs_var, prev_action_var])
        else:
            all_input_var = obs_var

        with tf.compat.v1.variable_scope(self._variable_scope):
            outputs, _, _, _, _, _ = self.model.build(
                all_input_var,
                self.model.networks['default'].step_input,
                self.model.networks['default'].step_hidden_input,
                self.model.networks['default'].step_cell_input,
                name=name)

        return dict(prob=outputs)

    def reset(self, dones=None):
        """Reset the policy."""
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self._prev_actions is None or len(dones) != len(self._prev_actions):
            self._prev_actions = np.zeros(
                (len(dones), self.action_space.flat_dim))
            self._prev_hiddens = np.zeros((len(dones), self._hidden_dim))
            self._prev_cells = np.zeros((len(dones), self._hidden_dim))

        self._prev_actions[dones] = 0.
        self._prev_hiddens[dones] = self.model.networks[
            'default'].init_hidden.eval()
        self._prev_cells[dones] = self.model.networks[
            'default'].init_cell.eval()

    def get_action(self, observation):
        """Return a single action."""
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        """Return multiple actions."""
        flat_obs = self.observation_space.flatten_n(observations)
        if self._state_include_action:
            assert self._prev_actions is not None
            all_input = np.concatenate([flat_obs, self._prev_actions], axis=-1)
        else:
            all_input = flat_obs
        probs, hidden_vec, cell_vec = self._f_step_prob(
            all_input, self._prev_hiddens, self._prev_cells)

        actions = list(map(self.action_space.weighted_sample, probs))
        prev_actions = self._prev_actions
        self._prev_actions = self.action_space.flatten_n(actions)
        self._prev_hiddens = hidden_vec
        self._prev_cells = cell_vec

        agent_info = dict(prob=probs)
        if self._state_include_action:
            agent_info['prev_action'] = np.copy(prev_actions)
        return actions, agent_info

    @property
    def recurrent(self):
        """Recurrent or not."""
        return True

    @property
    def distribution(self):
        """Policy distribution."""
        return RecurrentCategorical(self._action_dim)

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
        del new_dict['_f_step_prob']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        super().__setstate__(state)
        self._initialize()
