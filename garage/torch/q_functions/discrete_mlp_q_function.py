from torch import nn

from garage.core import Serializable
from garage.envs.util import flat_dim
from garage.torch.q_functions import QFunction


class DiscreteMLPQFunction(QFunction, Serializable):
    """
    This class implements a q value network to predict q based on the input
    state and action. It uses an MLP to fit the function of Q(s, a).
    """

    def __init__(self,
                 env_spec,
                 name="DiscreteMLPQFunction",
                 hidden_sizes=(32, 32),
                 network=None,
                 hidden_nonlinearity=nn.PReLU,
                 output_nonlinearity=None,
                 input_include_goal=False,
                 bn=False):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec():
            name(str, optional): A str contains the name of the policy.
            hidden_sizes(list or tuple, optional):
                A list of numbers of hidden units for all hidden layers.
            hidden_nonlinearity(optional):
                An activation shared by all fc layers.
            output_nonlinearity(optional):
                An activation used by the output layer.
            bn(bool, optional):
                A bool to indicate whether normalize the layer or not.
        """
        super().__init__()
        Serializable.quick_init(self, locals())

        self.name = name
        self._env_spec = env_spec
        if input_include_goal:
            obs_dim = flat_dim(
                env_spec.observation_space.spaces["observation"])
            goal_dim = flat_dim(
                env_spec.observation_space.spaces["desired_goal"])
            self._obs_dim = obs_dim + goal_dim
        else:
            self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._batch_norm = bn
        if network is None:
            self.model = None
            self._build_net()
        else:
            self.model = network

    def _build_net(self):
        layers = [nn.Linear(self._obs_dim, self._hidden_sizes[0])]
        for prev_size, curr_size in zip(self._hidden_sizes,
                                        self._hidden_sizes[1:]):
            layers.append(self._hidden_nonlinearity())
            layers.append(nn.Linear(prev_size, curr_size))
        layers.append(nn.Linear(self._hidden_sizes[-1], self._action_dim))
        if self._output_nonlinearity is not None:
            layers.append(self._output_nonlinearity)
        self.model = nn.Sequential(*layers)

    def get_qval(self, observation):
        return self.model.forward(observation)

    def get_params_internal(self, **tags):
        return self.model.parameters()
