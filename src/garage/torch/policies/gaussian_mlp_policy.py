"""GaussianMLPPolicy."""
import torch
from torch import nn

from garage.torch.modules import GaussianMLPModule
from garage.torch.policies import Policy


class GaussianMLPPolicy(Policy, GaussianMLPModule):
    """GaussianMLPPolicy.

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): Minimum value for std.
        max_std (float): Maximum value for std.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 name='GaussianMLPPolicy'):
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        Policy.__init__(self, env_spec, name)
        GaussianMLPModule.__init__(self,
                                   input_dim=self._obs_dim,
                                   output_dim=self._action_dim,
                                   hidden_sizes=hidden_sizes,
                                   hidden_nonlinearity=hidden_nonlinearity,
                                   hidden_w_init=hidden_w_init,
                                   hidden_b_init=hidden_b_init,
                                   output_nonlinearity=output_nonlinearity,
                                   output_w_init=output_w_init,
                                   output_b_init=output_b_init,
                                   learn_std=learn_std,
                                   init_std=init_std,
                                   min_std=min_std,
                                   max_std=max_std,
                                   std_parameterization=std_parameterization,
                                   layer_normalization=layer_normalization)

    def get_action(self, observation):
        """Get a single action given an observation.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            tuple:
                * np.ndarray: Predicted action.
                * dict:
                    * list[float]: Mean of the distribution
                    * list[float]: Standard deviation of logarithmic values of
                        the distribution

        """
        with torch.no_grad():
            observation = torch.Tensor(observation).unsqueeze(0)
            dist = self.forward(observation)
            return (dist.rsample().squeeze(0).numpy(),
                    dict(mean=dist.mean.squeeze(0).numpy(),
                         log_std=(dist.variance**.5).log().squeeze(0).numpy()))

    def get_actions(self, observations):
        """Get actions given observations.

        Args:
            observations (np.ndarray): Observations from the environment.

        Returns:
            tuple:
                * np.ndarray: Predicted actions.
                * dict:
                    * list[float]: Mean of the distribution
                    * list[float]: Standard deviation of logarithmic values of
                        the distribution

        """
        with torch.no_grad():
            dist = self.forward(torch.Tensor(observations))
            return (dist.rsample().numpy(),
                    dict(mean=dist.mean.numpy(),
                         log_std=(dist.variance**.5).log().numpy()))

    def log_likelihood(self, observation, action):
        """Compute log likelihood given observations and action.

        Args:
            observation (torch.Tensor): Observation from the environment.
            action (torch.Tensor): Predicted action.

        Returns:
            torch.Tensor: Calculated log likelihood value of the action given
                observation

        """
        dist = self.forward(observation)
        return dist.log_prob(action)

    def entropy(self, observation):
        """Get entropy given observations.

        Args:
            observation (torch.Tensor): Observation from the environment.

        Returns:
             torch.Tensor: Calculated entropy values given observation

        """
        dist = self.forward(observation)
        return dist.entropy()

    def reset(self, dones=None):
        """Reset the environment.

        Args:
            dones (numpy.ndarray): Reset values

        """

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            bool: flag for vectorized

        """
        return True
