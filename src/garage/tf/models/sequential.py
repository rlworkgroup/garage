"""Sequential Model.

A model composed of one or more models which are connected sequential,
according to the insertion order.
"""
from garage.tf.models.model import Model


class Sequential(Model):
    """Sequential Model.

    Args:
        name (str): Model name, also the variable scope.
        models (list[garage.tf.models.Model]): The models to be connected
            in sequential order.

    """

    def __init__(self, *models, name=None):
        super().__init__(name)
        self._models = models
        self._first_network = None
        self._last_network = None

    # pylint: disable=arguments-differ
    def _build(self, input_var, name=None):
        """Build model given input placeholder(s).

        Args:
            input_var (tf.Tensor): Tensor input.
            name (str): Inner model name, also the variable scope of the
                inner model.

        Return:
            tf.Tensor: Tensor output of the model.

        """
        out = input_var
        for model in self._models:
            self._last_network = model.build(out, name=name)
            if self._first_network is None:
                self._first_network = self._last_network
            out = self._last_network.outputs

        return out

    @property
    def input(self):
        """tf.Tensor: input of the model by default."""
        return self._first_network.input

    @property
    def output(self):
        """tf.Tensor: output of the model by default."""
        return self._last_network.output

    @property
    def inputs(self):
        """tf.Tensor: inputs of the model by default."""
        return self._first_network.inputs

    @property
    def outputs(self):
        """tf.Tensor: outputs of the model by default."""
        return self._last_network.outputs

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_first_network']
        del new_dict['_last_network']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._first_network = None
        self._last_network = None
