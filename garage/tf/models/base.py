"""
This file contains the abstraction class for models
"""
from garage.tf.models import AutoPickable

# flake8: noqa


class Model(AutoPickable):
    """
    Abstraction class for models.
    """

    def build_model(self):
        """
        This function will build the whole graph for
        the model. All tensors should be created here.
        By calling this function, a copy of the same graph
        should be created with the same parameters.
        Currently three things are returned: inputs, outputs and
        model_info. All of them are returned as dictionaries.
        """
        raise NotImplementedError

    @property
    def input(self):
        if len(self.model.inputs) == 1:
            return self.model.inputs[0]

        return self.model.inputs

    @property
    def output(self):
        if len(self.model.outputs) == 1:
            return self.model.outputs[0]

        return self.model.outputs

    @property
    def inputs(self):
        """
        Tensors of the inputs
        """
        return self.model.inputs

    @property
    def outputs(self):
        """
        Tensors of the outputs
        """
        return self.model.outputs
