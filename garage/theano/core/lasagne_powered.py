import lasagne.layers as L

from garage.core import Parameterized
from garage.misc.overrides import overrides


class LasagnePowered(Parameterized):
    def __init__(self, output_layers):
        self._output_layers = output_layers
        super(LasagnePowered, self).__init__()

    @property
    def output_layers(self):
        return self._output_layers

    # this gives ALL the vars (not the params values)
    @overrides
    def get_params_internal(self, **tags):
        # this lasagne function also returns all var below the passed layers
        return L.get_all_params(L.concat(self._output_layers), **tags)
