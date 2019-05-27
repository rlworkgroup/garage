"""
Sequential Model.

A model composed of one or more models which are connected sequential,
according to the insertion order.
"""
from garage.tf.models.base import Model


class Sequential(Model):
    """
    Sequential Model.

    Args:
        name (str): Model name, also the variable scope.
        models (list[garage.tf.models.Model]): The models to be connected
            in sequential order.
    """

    def __init__(self, *models, name=None):
        super().__init__(name)
        self._models = models

    def _build(self, input_var, name=None):
        out = input_var
        for model in self._models:
            out = model.build(out, name=name)

        return out

    @property
    def input(self):
        """tf.Tensor input of the model by default."""
        return self._models[0].networks['default'].input

    @property
    def output(self):
        """tf.Tensor output of the model by default."""
        return self._models[-1].networks['default'].output

    @property
    def inputs(self):
        """tf.Tensor inputs of the model by default."""
        return self._models[0].networks['default'].inputs

    @property
    def outputs(self):
        """tf.Tensor outputs of the model by default."""
        return self._models[-1].networks['default'].outputs
