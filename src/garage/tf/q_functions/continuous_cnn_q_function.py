"""Continuous CNN QFunction with CNN-MLP structure."""
import akro
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models import CNNMLPMergeModel


class ContinuousCNNQFunction(CNNMLPMergeModel):
    """Q function based on a CNN-MLP structure for continuous action space.

    This class implements a Q value network to predict Q based on the
    input state and action. It uses an CNN and a MLP to fit the function
    of Q(s, a).

    Args:
        env_spec (EnvSpec): Environment specification.
        filters (Tuple[Tuple[int, Tuple[int, int]], ...]): Number and dimension
            of filters. For example, ((3, (3, 5)), (32, (3, 3))) means there
            are two convolutional layers. The filter for the first layer have 3
            channels and its shape is (3 x 5), while the filter for the second
            layer have 32 channels and its shape is (3 x 3).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_sizes (tuple[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this q-function consists of
            two hidden layers, each with 32 hidden units.
        action_merge_layer (int): The index of layers at which to concatenate
            action inputs with the network. The indexing works like standard
            python list indexing. Index of 0 refers to the input layer
            (observation input) while an index of -1 points to the last
            hidden layer. Default parameter points to second layer from the
            end.
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
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 env_spec,
                 filters,
                 strides,
                 hidden_sizes=(256, ),
                 action_merge_layer=-2,
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
                 layer_normalization=False):

        if (not isinstance(env_spec.observation_space, akro.Box)
                or not len(env_spec.observation_space.shape) in (2, 3)):
            raise ValueError(
                '{} can only process 2D, 3D akro.Image or'
                ' akro.Box observations, but received an env_spec with '
                'observation_space of type {} and shape {}'.format(
                    type(self).__name__,
                    type(env_spec.observation_space).__name__,
                    env_spec.observation_space.shape))

        self._env_spec = env_spec
        self._filters = filters
        self._strides = strides
        self._hidden_sizes = hidden_sizes
        self._action_merge_layer = action_merge_layer
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

        self._obs_dim = self._env_spec.observation_space.shape
        self._action_dim = self._env_spec.action_space.shape

        super().__init__(name=name,
                         filters=self._filters,
                         strides=self._strides,
                         hidden_sizes=self._hidden_sizes,
                         action_merge_layer=self._action_merge_layer,
                         padding=self._padding,
                         max_pooling=self._max_pooling,
                         pool_strides=self._pool_strides,
                         pool_shapes=self._pool_shapes,
                         cnn_hidden_nonlinearity=self._cnn_hidden_nonlinearity,
                         hidden_nonlinearity=self._hidden_nonlinearity,
                         hidden_w_init=self._hidden_w_init,
                         hidden_b_init=self._hidden_b_init,
                         output_nonlinearity=self._output_nonlinearity,
                         output_w_init=self._output_w_init,
                         output_b_init=self._output_b_init,
                         layer_normalization=self._layer_normalization)

        self._initialize()

    def _initialize(self):

        action_ph = tf.compat.v1.placeholder(tf.float32,
                                             (None, ) + self._action_dim,
                                             name='action')
        if isinstance(self._env_spec.observation_space, akro.Image):
            obs_ph = tf.compat.v1.placeholder(tf.uint8,
                                              (None, ) + self._obs_dim,
                                              name='state')
            augmented_obs_ph = tf.cast(obs_ph, tf.float32) / 255.0
        else:
            obs_ph = tf.compat.v1.placeholder(tf.float32,
                                              (None, ) + self._obs_dim,
                                              name='state')
            augmented_obs_ph = obs_ph
        outputs = super().build(augmented_obs_ph, action_ph).outputs
        self._f_qval = tf.compat.v1.get_default_session().make_callable(
            outputs, feed_list=[obs_ph, action_ph])

        self._obs_input = obs_ph
        self._act_input = action_ph

    @property
    def inputs(self):
        """tuple[tf.Tensor]: The observation and action input tensors.

        The returned tuple contains two tensors. The first is the observation
        tensor with shape :math:`(N, O*)`, and the second is the action tensor
        with shape :math:`(N, A*)`.
        """
        return self._obs_input, self._act_input

    def get_qval(self, observation, action):
        """Q Value of the network.

        Args:
            observation (np.ndarray): Observation input of shape
                :math:`(N, O*)`.
            action (np.ndarray): Action input of shape :math:`(N, A*)`.

        Returns:
            np.ndarray: Array of shape :math:`(N, )` containing Q values
                corresponding to each (obs, act) pair.

        """
        if len(observation[0].shape) < len(self._obs_dim):
            observation = self._env_spec.observation_space.unflatten_n(
                observation)

        return self._f_qval(observation, action)

    # pylint: disable=arguments-differ
    def build(self, state_input, action_input, name):
        """Build the symbolic graph for q-network.

        Args:
            state_input (tf.Tensor): The state input tf.Tensor of shape
                :math:`(N, O*)`.
            action_input (tf.Tensor): The action input tf.Tensor of shape
                :math:`(N, A*)`.
            name (str): Network variable scope.

        Return:
            tf.Tensor: The output Q value tensor of shape :math:`(N, )`.

        """
        augmented_state_input = state_input
        if isinstance(self._env_spec.observation_space, akro.Image):
            augmented_state_input = tf.cast(state_input, tf.float32) / 255.0
        return super().build(augmented_state_input, action_input,
                             name=name).outputs

    def clone(self, name):
        """Return a clone of the Q-function.

        It copies the configuration of the primitive and also the parameters.

        Args:
            name (str): Name of the newly created q-function.

        Return:
            ContinuousCNNQFunction: Cloned Q function.

        """
        new_qf = self.__class__(
            name=name,
            env_spec=self._env_spec,
            filters=self._filters,
            strides=self._strides,
            hidden_sizes=self._hidden_sizes,
            action_merge_layer=self._action_merge_layer,
            padding=self._padding,
            max_pooling=self._max_pooling,
            pool_shapes=self._pool_shapes,
            pool_strides=self._pool_strides,
            cnn_hidden_nonlinearity=self._cnn_hidden_nonlinearity,
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
        del new_dict['_obs_input']
        del new_dict['_act_input']
        return new_dict

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        super().__setstate__(state)
        self._initialize()
