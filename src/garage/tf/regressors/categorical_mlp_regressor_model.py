"""CategoricalMLPRegressorModel."""
import tensorflow as tf
import tensorflow_probability as tfp

from garage.experiment import deterministic
from garage.tf.models import NormalizedInputMLPModel


class CategoricalMLPRegressorModel(NormalizedInputMLPModel):
    """CategoricalMLPRegressorModel based on garage.tf.models.Model class.

    This class can be used to perform regression by fitting a Categorical
    distribution to the outputs.

    Args:
        input_shape (tuple[int]): Input shape of the training data.
        output_dim (int): Output dimension of the model.
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
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 input_shape,
                 output_dim,
                 name='CategoricalMLPRegressorModel',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False):
        super().__init__(input_shape=input_shape,
                         output_dim=output_dim,
                         name=name,
                         hidden_sizes=hidden_sizes,
                         hidden_nonlinearity=hidden_nonlinearity,
                         hidden_w_init=hidden_w_init,
                         hidden_b_init=hidden_b_init,
                         output_nonlinearity=output_nonlinearity,
                         output_w_init=output_w_init,
                         output_b_init=output_b_init,
                         layer_normalization=layer_normalization)

    def network_output_spec(self):
        """Network output spec.

        Return:
            list[str]: List of key(str) for the network outputs.

        """
        return ['y_hat', 'x_mean', 'x_std', 'dist']

    def _build(self, state_input, name=None):
        """Build model.

        Args:
            state_input (tf.Tensor): Observation inputs.
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Returns:
            tf.Tensor: Tensor output of the model.
            tf.Tensor: Mean for data.
            tf.Tensor: log_std for data.
            tfp.distributions.OneHotCategorical: Categorical distribution.

        """
        y_hat, x_mean_var, x_std_var = super()._build(state_input, name=name)
        dist = tfp.distributions.OneHotCategorical(probs=y_hat)
        return y_hat, x_mean_var, x_std_var, dist

    def clone(self, name):
        """Return a clone of the model.

        It only copies the configuration of the primitive,
        not the parameters.

        Args:
            name (str): Name of the newly created model. It has to be
                different from source model if cloned under the same
                computational graph.

        Returns:
            garage.tf.regressors.CategoricalMLPRegressorModel: Newly cloned
                model.

        """
        return self.__class__(name=name,
                              input_shape=self._input_shape,
                              output_dim=self._output_dim,
                              hidden_sizes=self._hidden_sizes,
                              hidden_nonlinearity=self._hidden_nonlinearity,
                              hidden_w_init=self._hidden_w_init,
                              hidden_b_init=self._hidden_b_init,
                              output_nonlinearity=self._output_nonlinearity,
                              output_w_init=self._output_w_init,
                              output_b_init=self._output_b_init,
                              layer_normalization=self._layer_normalization)
