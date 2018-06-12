import tensorflow as tf

from rllab.core import Serializable
from rllab.misc import ext
from rllab.tf.core import LayersPowered
from rllab.tf.core import MLP
import rllab.tf.core.layers as L
from rllab.tf.core.layers import batch_norm
from rllab.tf.distributions import Categorical
from rllab.tf.misc import tensor_utils
from rllab.tf.misc import tensor_utils
from rllab.tf.misc.tensor_utils import enclosing_scope
from rllab.tf.policies import StochasticPolicy
from rllab.tf.q_functions import QFunction


class ContinuousMLPQFunction(QFunction, LayersPowered, Serializable):
    def __init__(self,
                 env_spec,
                 name="ContinuousMLPQFunction",
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 action_merge_layer=-2,
                 output_nonlinearity=None,
                 bn=False):
        Serializable.quick_init(self, locals())

        l_obs = L.InputLayer(
            shape=(None, env_spec.observation_space.flat_dim), name="obs")
        l_action = L.InputLayer(
            shape=(None, env_spec.action_space.flat_dim), name="actions")

        n_layers = len(hidden_sizes) + 1

        if n_layers > 1:
            action_merge_layer = \
                (action_merge_layer % n_layers + n_layers) % n_layers
        else:
            action_merge_layer = 1

        l_hidden = l_obs

        for idx, size in enumerate(hidden_sizes):
            if bn:
                l_hidden = batch_norm(l_hidden)

            if idx == action_merge_layer:
                l_hidden = L.ConcatLayer([l_hidden, l_action])

            l_hidden = L.DenseLayer(
                l_hidden,
                num_units=size,
                nonlinearity=hidden_nonlinearity,
                name="h%d" % (idx + 1))

        if action_merge_layer == n_layers:
            l_hidden = L.ConcatLayer([l_hidden, l_action])

        l_output = L.DenseLayer(
            l_hidden,
            num_units=1,
            nonlinearity=output_nonlinearity,
            name="output")

        output_var = L.get_output(l_output, deterministic=True)

        self._f_qval = tensor_utils.compile_function(
            [l_obs.input_var, l_action.input_var], output_var)
        self._output_layer = l_output
        self._obs_layer = l_obs
        self._action_layer = l_action
        self._output_nonlinearity = output_nonlinearity
        self._name = name

        LayersPowered.__init__(self, [l_output])

    def get_qval(self, observations, actions):
        return self._f_qval(observations, actions)

    def get_qval_sym(self, obs_var, action_var, name="get_qval_sym", **kwargs):
        with enclosing_scope(self._name, name):
            qvals = L.get_output(self._output_layer, {
                self._obs_layer: obs_var,
                self._action_layer: action_var
            }, **kwargs)
            return tf.reshape(qvals, (-1, ))
