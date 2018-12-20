"""This file contains the tautology for garage models."""
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel

from garage.misc.overrides import overrides
from garage.tf.core import Parameterized


class Model(KerasModel, Parameterized):
    """
    This is a tautology for all the garage models.

    A model define a neural network model to build a graph for
    other primitivs (e.g. GaussianMLPPolicy and GaussianMLPBaseline).
    A model can be built by assembling different models. For example,
    to define a VAEGaussianMLP Model, we need to build a VAE model
    and a GaussianMLP Model. Then, we can assemble them in the
    build_model function.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a model."""
        super(Model, self).__init__(*args, **kwargs)
        Parameterized.__init__(self)

    def build_model(self, inputs=None, reuse=tf.AUTO_REUSE):
        """
        Build the whole graph for the model.

        All tensors should be created here. By calling this function,
        a copy of the same graph should be created with the same
        parameters. Currently three things are returned: inputs,
        outputs and model_info. All of them are returned as dictionaries.
        """
        raise NotImplementedError

    @overrides
    def get_params_internal(self, **tags):
        """
        Retrieve the parameters to pickle the model.

        Args
            tags: a dictionary of tags, only support trainable for now.
        Return
            parameters of the model in tf.Tensor.
        """
        if tags.get("trainable"):
            params = [v for v in tf.trainable_variables(scope=self.name)]
        else:
            params = [v for v in tf.global_variables(scope=self.name)]
        return params
