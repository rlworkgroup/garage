"""This file contains the abstraction class for models."""
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.core import Parameterized


class Model(Parameterized):
    """Abstraction class for models."""

    def __init__(self):
        """Initialize a model."""
        Parameterized.__init__(self)

    def build_model(self):
        """
        Build the whole graph for the model.

        All tensors should be created here. By calling this function,
        a copy of the same graph should be created with the same
        parameters. Currently three things are returned: inputs,
        outputs and model_info. All of them are returned as dictionaries.
        """
        raise NotImplementedError

    @property
    def inputs(self, inputs=None, reuse=tf.AUTO_REUSE):
        """Tensors of the inputs."""
        return self._inputs

    @property
    def outputs(self):
        """Tensors of the outputs."""
        return self._outputs

    @overrides
    def get_params_internal(self, **tags):
        """
        Retrieve the parameters to pickle the model.

        Args
            tags: a dictionary of tags, only support trainable for now.
        Return
            parameters of the model.
        """
        if tags.get("trainable"):
            params = [v for v in tf.trainable_variables(scope=self.name)]
        else:
            params = [v for v in tf.global_variables(scope=self.name)]
        return params
