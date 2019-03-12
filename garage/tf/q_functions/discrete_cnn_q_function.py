"""Discrete MLP QFunction."""
import tensorflow as tf

from garage.tf.models.cnn_model import CNNModel
from garage.tf.models.cnn_model_max_pooling import CNNModelWithMaxPooling
from garage.tf.models.discrete_mlp_dueling_model import DiscreteMLPDuelingModel
from garage.tf.models.discrete_mlp_model import DiscreteMLPModel
from garage.tf.q_functions import QFunction2


class DiscreteCNNQFunction(QFunction2):
    """
    Q function based on CNN for discrete action space.

    This class implements a Q value network to predict Q based on the
    input state and action. It uses an CNN to fit the function of Q(s, a).

    Args:
        env_spec: environment specification
        filter_dims: Dimension of the filters.
        num_filters: Number of filters.
        strides: The strides of the sliding window.
        hidden_sizes: Output dimension of dense layer(s).
        name: Variable scope of the cnn.
        padding: The type of padding algorithm to use, from "SAME", "VALID".
        max_pooling: Boolean for using max pooling layer or not.
        pool_shape: Dimension of the pooling layer(s).
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        output_w_init: Initializer function for the weight
                    of output dense layer(s).
        output_b_init: Initializer function for the bias
                    of output dense layer(s).
        dueling: Bool for using dueling network or not.
        layer_normalization: Bool for using layer normalization or not.
    """

    def __init__(self,
                 env_spec,
                 filter_dims,
                 num_filters,
                 strides,
                 hidden_sizes=[256],
                 name="DiscreteCNNQFunction",
                 padding="SAME",
                 max_pooling=False,
                 pool_strides=(2, 2),
                 pool_shapes=(2, 2),
                 cnn_hidden_nonlinearity=tf.nn.relu,
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.contrib.layers.xavier_initializer,
                 hidden_b_init=tf.zeros_initializer,
                 output_nonlinearity=None,
                 output_w_init=tf.contrib.layers.xavier_initializer,
                 output_b_init=tf.zeros_initializer,
                 dueling=False,
                 layer_normalization=False):
        super().__init__(name)
        self._env_spec = env_spec
        self._action_dim = env_spec.action_space.n
        self._filter_dims = filter_dims
        self._num_filters = num_filters
        self._strides = strides
        self._hidden_sizes = hidden_sizes
        self._padding = padding
        self._max_pooling = max_pooling
        self._pool_strides = pool_strides
        self._pool_shapes = pool_shapes
        self._cnn_hidden_nonlinearity = cnn_hidden_nonlinearity
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._dueling = dueling

        obs_dim = self._env_spec.observation_space.shape
        action_dim = env_spec.action_space.flat_dim

        self.models = []

        if not max_pooling:
            cnn_model = CNNModel(
                filter_dims=filter_dims,
                num_filters=num_filters,
                strides=strides,
                padding=padding,
                hidden_nonlinearity=cnn_hidden_nonlinearity)
        else:
            cnn_model = CNNModelWithMaxPooling(
                filter_dims=filter_dims,
                num_filters=num_filters,
                strides=strides,
                padding=padding,
                pool_strides=pool_strides,
                pool_shapes=pool_shapes,
                hidden_nonlinearity=cnn_hidden_nonlinearity)
        self.models.append(cnn_model)
        if not dueling:
            output_model = DiscreteMLPModel(
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                hidden_w_init=hidden_w_init,
                hidden_b_init=hidden_b_init,
                output_nonlinearity=output_nonlinearity,
                output_w_init=output_w_init,
                output_b_init=output_b_init,
                layer_normalization=layer_normalization)
        else:
            output_model = DiscreteMLPDuelingModel(
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                hidden_w_init=hidden_w_init,
                hidden_b_init=hidden_b_init,
                output_nonlinearity=output_nonlinearity,
                output_w_init=output_w_init,
                output_b_init=output_b_init,
                layer_normalization=layer_normalization)
        self.models.append(output_model)

        obs_ph = tf.placeholder(tf.float32, (None, ) + obs_dim, name="obs")

        with tf.variable_scope(self._variable_scope):
            out = obs_ph
            for model in self.models:
                out = model.build(out)

    def q_vals(self):
        """Q values."""
        return self.models[-1].networks['default'].outputs

    @property
    def input(self):
        """Input tf.Tensor of the Q-function."""
        return self.models[0].networks['default'].input

    def get_qval_sym(self, state_input, name):
        """
        Symbolic graph for q-network.

        Args:
            state_input: The state input tf.Tensor to the network.
            name: Network variable scope.

        Return:
            The tf.Tensor output of Discrete CNN QFunction.
        """
        with tf.variable_scope(self._variable_scope):
            out = state_input
            for model in self.models:
                out = model.build(out, name=name)

            return out

    def clone(self, name):
        """
        Return a clone of the Q-function.

        Args:
            name: Name of the newly created q-function.
        """
        return self.__class__(
            name=name,
            env_spec=self._env_spec,
            filter_dims=self._filter_dims,
            num_filters=self._num_filters,
            strides=self._strides,
            hidden_sizes=self._hidden_sizes,
            padding=self._padding,
            max_pooling=self._max_pooling,
            pool_shapes=self._pool_shapes,
            pool_strides=self._pool_strides,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            dueling=self._dueling,
            layer_normalization=self._layer_normalization)
