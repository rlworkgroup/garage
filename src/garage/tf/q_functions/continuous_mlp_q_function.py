"""Continuous MLP QFunction."""
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models import MLPMergeModel


class ContinuousMLPQFunction(MLPMergeModel):
    """Continuous MLP QFunction.

    This class implements a q value network to predict q based on the input
    state and action. It uses an MLP to fit the function of Q(s, a).

    Args:
        env_spec (EnvSpec): Environment specification.
        name (str): Name of the q-function, also serves as the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this q-function consists of
            two hidden layers, each with 32 hidden units.
        action_merge_layer (int): The index of layers at which to concatenate
            action inputs with the network. The indexing works like standard
            python list indexing. Index of 0 refers to the input layer
            (observation input) while an index of -1 points to the last
            hidden layer. Default parameter points to second layer from the
            end.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
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
        layer_normalization (bool): Bool for using layer normalization.

    """

    def __init__(self,
                 env_spec,
                 name='ContinuousMLPQFunction',
                 hidden_sizes=(32, 32),
                 action_merge_layer=-2,
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False):

        self._env_spec = env_spec
        self._hidden_sizes = hidden_sizes
        self._action_merge_layer = action_merge_layer
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        super().__init__(name=name,
                         output_dim=1,
                         hidden_sizes=hidden_sizes,
                         concat_layer=self._action_merge_layer,
                         hidden_nonlinearity=hidden_nonlinearity,
                         hidden_w_init=hidden_w_init,
                         hidden_b_init=hidden_b_init,
                         output_nonlinearity=output_nonlinearity,
                         output_w_init=output_w_init,
                         output_b_init=output_b_init,
                         layer_normalization=layer_normalization)
        self._network = None

        self._initialize()

    def _initialize(self):
        obs_ph = tf.compat.v1.placeholder(tf.float32, (None, self._obs_dim),
                                          name='obs')
        action_ph = tf.compat.v1.placeholder(tf.float32,
                                             (None, self._action_dim),
                                             name='act')

        self._network = super().build(obs_ph, action_ph)

        self._f_qval = tf.compat.v1.get_default_session().make_callable(
            self._network.outputs, feed_list=[obs_ph, action_ph])

    def get_qval(self, observation, action):
        """Q Value of the network.

        Args:
            observation (np.ndarray): Observation input.
            action (np.ndarray): Action input.

        Returns:
            np.ndarray: Q values.

        """
        return self._f_qval(observation, action)

    @property
    def inputs(self):
        """Return the input tensor.

        Returns:
            tf.Tensor: The input tensors of the model.

        """
        return self._network.inputs

    # pylint: disable=arguments-differ
    def build(self, state_input, action_input, name):
        """Build the symbolic graph for q-network.

        Args:
            state_input (tf.Tensor): The state input tf.Tensor to the network.
            action_input (tf.Tensor): The action input tf.Tensor to the
                network.
            name (str): Network variable scope.

        Return:
            tf.Tensor: The output of Continuous MLP QFunction.

        """
        return super().build(state_input, action_input, name=name).outputs

    def clone(self, name):
        """Return a clone of the Q-function.

        It copies the configuration of the primitive and also the parameters.

        Args:
            name (str): Name of the newly created q-function.

        Returns:
            ContinuousMLPQFunction: A new instance with same arguments.

        """
        new_qf = self.__class__(name=name,
                                env_spec=self._env_spec,
                                hidden_sizes=self._hidden_sizes,
                                action_merge_layer=self._action_merge_layer,
                                hidden_nonlinearity=self._hidden_nonlinearity,
                                hidden_w_init=self._hidden_w_init,
                                hidden_b_init=self._hidden_b_init,
                                output_nonlinearity=self._output_nonlinearity,
                                output_w_init=self._output_w_init,
                                output_b_init=self._output_b_init,
                                layer_normalization=self._layer_normalization)
        new_qf.parameters = self.parameters
        return new_qf

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_qval']
        del new_dict['_network']
        return new_dict

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        super().__setstate__(state)
        self._initialize()
