"""GaussianMLPEmbedding."""
import akro
import numpy as np
import tensorflow as tf

from garage.experiment.deterministic import get_seed
from garage.tf.embeddings import StochasticEmbedding
from garage.tf.models import GaussianMLPModel


class GaussianMLPEmbedding(StochasticEmbedding):
    """GaussianMLPEmbedding with GaussianMLPModel.

    An embedding that contains a MLP to make prediction based on
    a gaussian distribution.

    Args:
        embedding (garage.tf.embeddings.embedding_spec.EmbeddingSpec):
            Embedding specification.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
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
        learn_std (bool): Is std trainable.
        adaptive_std (bool): Is std a neural network. If False, it will be a
            parameter.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        init_std (float): Initial value for std.
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues.
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues.
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        std_output_nonlinearity (callable): Nonlinearity for output layer in
            the std network. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        std_parameterization (str): How the std should be parametrized. There
            are a few options:
            - exp: the logarithm of the std will be stored, and applied a
                exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 embedding_spec,
                 name="GaussianMLPEmbedding",
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 learn_std=True,
                 adaptive_std=False,
                 std_share_network=False,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=tf.nn.tanh,
                 std_output_nonlinearity=None,
                 std_parameterization='exp',
                 layer_normalization=False):
        assert isinstance(embedding_spec.latent_space, akro.Box)
        super().__init__(name, embedding_spec)
        self.latent_dim = embedding_spec.latent_space.flat_dim
        self.input_dim = embedding_spec.input_space.flat_dim

        self.model = GaussianMLPModel(
            output_dim=self.latent_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            adaptive_std=adaptive_std,
            std_share_network=std_share_network,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_hidden_sizes=std_hidden_sizes,
            std_hidden_nonlinearity=std_hidden_nonlinearity,
            std_output_nonlinearity=std_output_nonlinearity,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization,
            name='GaussianMLPModel')

        self._initialize()

    def _initialize(self):
        embedding_input = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.input_dim))

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            self.model.build(embedding_input)

        self._f_dist = tf.compat.v1.get_default_session().make_callable(
            [
                self.model.networks['default'].mean,
                self.model.networks['default'].log_std
            ],
            feed_list=[self.model.networks['default'].input])

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            Bool: True if primitive supports vectorized operations.

        """
        return True

    def dist_info_sym(self, in_var, state_info_vars=None, name='default'):
        """Build a symbolic graph of the distribution parameters.

        Args:
            in_var (tf.Tensor): Tensor input for symbolic graph.
            state_info_vars (dict): Extra state information, e.g.
                previous action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of distribution
                parameters.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            mean_var, log_std_var, _, _ = self.model.build(in_var, name=name)
        return dict(mean=mean_var, log_std=log_std_var)

    def get_latent(self, given):
        """Get single latent from this policy for the given input.

        Args:
            given (numpy.ndarray): Input from environment.

        Returns:
            numpy.ndarray: Latent samples.
            dict: Predicted action and agent information.

        Note:
            It returns an action and a dict, with keys
            - mean (numpy.ndarray): Mean of the distribution.
            - log_std (numpy.ndarray): Log standard deviation of the
                distribution.

        """
        flat_input = self.input_space.flatten(given)
        mean, log_std = self._f_dist([flat_input])
        rnd = np.random.normal(size=mean.shape)
        sample = rnd * np.exp(log_std) + mean
        sample = self.latent_space.unflatten(sample[0])
        mean = self.latent_space.unflatten(mean[0])
        log_std = self.latent_space.unflatten(log_std[0])
        return sample, dict(mean=mean, log_std=log_std)

    def get_latents(self, inputs):
        """Get multiple latents from this policy for the inputs.

        Args:
            inputs (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Actions
            dict: Predicted action and agent information.

        Note:
            It returns actions and a dict, with keys
            - mean (numpy.ndarray): Means of the distribution.
            - log_std (numpy.ndarray): Log standard deviations of the
                distribution.

        """
        flat_inputs = self.input_space.flatten_n(inputs)
        means, log_stds = self._f_dist(flat_inputs)
        rnd = np.random.normal(size=means.shape)
        samples = rnd * np.exp(log_stds) + means
        samples = self.latent_space.unflatten_n(samples)
        means = self.latent_space.unflatten_n(means)
        log_stds = self.latent_space.unflatten_n(log_stds)
        return samples, dict(mean=means, log_std=log_stds)

    def get_params(self):
        """Get the params, which are the trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        return self.get_trainable_vars()

    @property
    def distribution(self):
        """Policy distribution.

        Returns:
            garage.tf.distributions.DiagonalGaussian: Policy distribution.

        """
        return self.model.networks['default'].dist

    @property
    def input(self):
        return self.model.networks['default'].input

    @property
    def latent_mean(self):
        return self.model.networks['default'].mean    

    @property
    def latent_std_param(self):
        return self.model.networks['default'].log_std
    
    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_dist']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
