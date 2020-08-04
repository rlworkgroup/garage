"""Discrete CNN QFunction with CNN-MLP structure."""
# yapf: disable
import akro
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models import (CNNModel,
                              CNNModelWithMaxPooling,
                              MLPDuelingModel,
                              MLPModel,
                              Sequential)

# yapf: enable


class DiscreteCNNQFunction(Sequential):
    """Q function based on a CNN-MLP structure for discrete action space.

    This class implements a Q value network to predict Q based on the
    input state and action. It uses an CNN and a MLP to fit the function
    of Q(s, a).

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        filters (Tuple[Tuple[int, Tuple[int, int]], ...]): Number and dimension
            of filters. For example, ((3, (3, 5)), (32, (3, 3))) means there
            are two convolutional layers. The filter for the first layer have 3
            channels and its shape is (3 x 5), while the filter for the second
            layer have 32 channels and its shape is (3 x 3).
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
                 filters,
                 strides,
                 hidden_sizes=(256, ),
                 name=None,
                 padding='SAME',
                 max_pooling=False,
                 pool_strides=(2, 2),
                 pool_shapes=(2, 2),
                 cnn_hidden_nonlinearity=tf.nn.relu,
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 dueling=False,
                 layer_normalization=False):
        if not isinstance(env_spec.observation_space, akro.Box) or \
                not len(env_spec.observation_space.shape) in (2, 3):
            raise ValueError(
                '{} can only process 2D, 3D akro.Image or'
                ' akro.Box observations, but received an env_spec with '
                'observation_space of type {} and shape {}'.format(
                    type(self).__name__,
                    type(env_spec.observation_space).__name__,
                    env_spec.observation_space.shape))

        self._env_spec = env_spec
        self._action_dim = env_spec.action_space.n
        self._filters = filters
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
            cnn_model = CNNModel(filters=filters,
                                 strides=strides,
                                 padding=padding,
                                 hidden_nonlinearity=cnn_hidden_nonlinearity)
        else:
            cnn_model = CNNModelWithMaxPooling(
                filters=filters,
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

        super().__init__(cnn_model, output_model, name=name)
        self._network = None

        self._initialize()

    def _initialize(self):
        """Initialize QFunction."""
        if isinstance(self._env_spec.observation_space, akro.Image):
            obs_ph = tf.compat.v1.placeholder(tf.uint8,
                                              (None, ) + self.obs_dim,
                                              name='obs')
            augmented_obs_ph = tf.cast(obs_ph, tf.float32) / 255.0
        else:
            obs_ph = tf.compat.v1.placeholder(tf.float32,
                                              (None, ) + self.obs_dim,
                                              name='obs')
            augmented_obs_ph = obs_ph

        self._network = super().build(augmented_obs_ph)

        self._obs_input = obs_ph

    @property
    def q_vals(self):
        """Return the Q values, the output of the network.

        Return:
            list[tf.Tensor]: Q values.

        """
        return self._network.outputs

    @property
    def input(self):
        """Get input.

        Return:
            tf.Tensor: QFunction Input.

        """
        return self._obs_input

    # pylint: disable=arguments-differ
    def build(self, state_input, name):
        """Build the symbolic graph for q-network.

        Args:
            state_input (tf.Tensor): The state input tf.Tensor to the network.
            name (str): Network variable scope.

        Return:
            tf.Tensor: The tf.Tensor output of Discrete CNN QFunction.

        """
        augmented_state_input = state_input
        if isinstance(self._env_spec.observation_space, akro.Image):
            augmented_state_input = tf.cast(state_input, tf.float32) / 255.0
        return super().build(augmented_state_input, name=name).outputs

    def clone(self, name):
        """Return a clone of the Q-function.

        It copies the configuration of the primitive and also the parameters.

        Args:
            name(str) : Name of the newly created q-function.

        Returns:
            garage.tf.q_functions.DiscreteCNNQFunction: Clone of this object

        """
        new_qf = self.__class__(name=name,
                                env_spec=self._env_spec,
                                filters=self._filters,
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
        new_qf.parameters = self.parameters
        return new_qf

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state.

        """
        new_dict = super().__getstate__()
        del new_dict['_obs_input']
        del new_dict['_network']
        return new_dict
