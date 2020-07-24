"""A dummy algorithm fixture."""
from garage.np.algos import RLAlgorithm


class DummyAlgo(RLAlgorithm):  # pylint: disable=too-few-public-methods
    """Dummy algo for test.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.np.policies.Policy): Policy.
        baseline (garage.np.baselines.Baseline): The baseline.

    """

    def __init__(self, env_spec, policy, baseline):
        self.env_spec = env_spec
        self.policy = policy
        self.baseline = baseline
        self.discount = 0.9
        self.max_episode_length = 1
        self.n_samples = 10

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        See garage.np.algos.RLAlgorithm train().

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
