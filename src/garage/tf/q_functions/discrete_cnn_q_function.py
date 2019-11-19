"""Discrete CNN QFunction with CNN-MLP structure."""
import tensorflow as tf

from garage.tf.models import CNNModel
from garage.tf.models import CNNModelWithMaxPooling
from garage.tf.models import MLPDuelingModel
from garage.tf.models import MLPModel
from garage.tf.models import Sequential
from garage.tf.q_functions import QFunction


class DiscreteCNNQFunction(QFunction):
    """Q function based on a CNN-MLP structure for discrete action space.

    This class implements a Q value network to predict Q based on the
    input state and action. It uses an CNN and a MLP to fit the function
    of Q(s, a).

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        filter_dims (tuple[int]): Dimension of the filters. For example,
            (3, 5) means there are two convolutional layers. The filter for
            first layer is of dimension (3 x 3) and the second one is of
            dimension (5 x 5).
        num_filters (tuple[int]): Number of filters. For example, (3, 32) means
            there are two convolutional layers. The filter for the first layer
            has 3 channels and the second one with 32 channels.
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this q-function consists of
            two hidden layers, each with 32 hidden units.
        name (str): Variable scope of the cnn.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        max_pooling (bool): Boolean for using max pooling layer or not.
        pool_shapes (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            shape (2, 2).
        pool_strides (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        cnn_hidden_nonlinearity (callable): Activation function for
            intermediate dense layer(s) in the CNN. It should return a
            tf.Tensor. Set it to None to maintain a linear activation.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s) in the MLP. It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s) in the MLP. The function should
            return a tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s) in the MLP. The function should
            return a tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer in the MLP. It should return a tf.Tensor. Set it to None
            to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the MLP. The function should return
            a tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s) in the MLP. The function should return
            a tf.Tensor.
        dueling (bool): Bool for using dueling network or not.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 env_spec,
                 filter_dims,
                 num_filters,
                 strides,
                 hidden_sizes=[256],
                 name=None,
                 padding='SAME',
                 max_pooling=False,
                 pool_strides=(2, 2),
                 pool_shapes=(2, 2),
                 cnn_hidden_nonlinearity=tf.nn.relu,
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
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

        self.obs_dim = self._env_spec.observation_space.shape
        action_dim = self._env_spec.action_space.flat_dim

        if not max_pooling:
            cnn_model = CNNModel(filter_dims=filter_dims,
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
        if not dueling:
            output_model = MLPModel(output_dim=action_dim,
                                    hidden_sizes=hidden_sizes,
                                    hidden_nonlinearity=hidden_nonlinearity,
                                    hidden_w_init=hidden_w_init,
                                    hidden_b_init=hidden_b_init,
                                    output_nonlinearity=output_nonlinearity,
                                    output_w_init=output_w_init,
                                    output_b_init=output_b_init,
                                    layer_normalization=layer_normalization)
        else:
            output_model = MLPDuelingModel(
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                hidden_w_init=hidden_w_init,
                hidden_b_init=hidden_b_init,
                output_nonlinearity=output_nonlinearity,
                output_w_init=output_w_init,
                output_b_init=output_b_init,
                layer_normalization=layer_normalization)

        self.model = Sequential(cnn_model, output_model)

        self._initialize()

    def _initialize(self):
        obs_ph = tf.compat.v1.placeholder(tf.float32, (None, ) + self.obs_dim,
                                          name='obs')
        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            self.model.build(obs_ph)

    @property
    def q_vals(self):
        """Q values."""
        return self.model.networks['default'].outputs

    @property
    def input(self):
        """Input tf.Tensor of the Q-function."""
        return self.model.networks['default'].input

    def get_qval_sym(self, state_input, name):
        """Symbolic graph for q-network.

        Args:
            state_input (tf.Tensor): The state input tf.Tensor to the network.
            name (str): Network variable scope.

        Return:
            The tf.Tensor output of Discrete CNN QFunction.
        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            return self.model.build(state_input, name=name)

    def clone(self, name):
        """Return a clone of the Q-function.

        It only copies the configuration of the Q-function,
        not the parameters.

        Args:
            name: Name of the newly created q-function.
        """
        return self.__class__(name=name,
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

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__.update(state)
        self._initialize()
