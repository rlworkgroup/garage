"""This class implements OffPolicyRLAlgorithm for off-policy RL algorithms."""
import abc

from dowel import tabular

from garage import log_performance, TrajectoryBatch
from garage.np.algos import RLAlgorithm
from garage.sampler import OffPolicyVectorizedSampler
from garage.sampler.utils import rollout


class OffPolicyRLAlgorithm(RLAlgorithm):
    """This class implements OffPolicyRLAlgorithm for off-policy RL algorithms.

    Off-policy algorithms such as DQN, DDPG can inherit from it.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        qf (object): The q value network.
        replay_buffer (garage.replay_buffer.ReplayBuffer): Replay buffer.
        use_target (bool): Whether to use target.
        discount(float): Discount factor for the cumulative return.
        steps_per_epoch (int): Number of train_once calls per epoch.
        max_path_length (int): Maximum path length. The episode will
            terminate when length of trajectory reaches max_path_length.
        n_train_steps (int): Training steps.
        buffer_batch_size (int): Batch size for replay buffer.
        min_buffer_size (int): The minimum buffer size for replay buffer.
        rollout_batch_size (int): Roll out batch size.
        reward_scale (float): Reward scale.
        input_include_goal (bool): Whether input includes goal.
        smooth_return (bool): Whether to smooth the return.
        exploration_strategy
            (garage.np.exploration_strategies.ExplorationStrategy):
            Exploration strategy.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 qf,
                 replay_buffer,
                 use_target=False,
                 discount=0.99,
                 steps_per_epoch=20,
                 max_path_length=None,
                 n_train_steps=50,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 rollout_batch_size=1,
                 reward_scale=1.,
                 input_include_goal=False,
                 smooth_return=True,
                 exploration_strategy=None):
        self.env_spec = env_spec
        self.policy = policy
        self.qf = qf
        self.replay_buffer = replay_buffer
        self.steps_per_epoch = steps_per_epoch
        self.n_train_steps = n_train_steps
        self.buffer_batch_size = buffer_batch_size
        self.use_target = use_target
        self.discount = discount
        self.min_buffer_size = min_buffer_size
        self.rollout_batch_size = rollout_batch_size
        self.reward_scale = reward_scale
        self.evaluate = False
        self.input_include_goal = input_include_goal
        self.smooth_return = smooth_return
        self.max_path_length = max_path_length
        self.es = exploration_strategy

        self.sampler_cls = OffPolicyVectorizedSampler

        self.init_opt()

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in runner.step_epochs():
            for cycle in range(self.steps_per_epoch):
                runner.step_path = runner.obtain_samples(runner.step_itr)
                for path in runner.step_path:
                    path['rewards'] *= self.reward_scale
                last_return = self.train_once(runner.step_itr,
                                              runner.step_path)
                if cycle == 0 and self.evaluate:
                    log_performance(runner.step_itr,
                                    self._obtain_evaluation_samples(
                                        runner.get_env_copy()),
                                    discount=self.discount)
                    tabular.record('TotalEnvSteps', runner.total_env_steps)
                runner.step_itr += 1

        return last_return

    def log_diagnostics(self, paths):
        """Log diagnostic information on current paths.

        Args:
            paths (list[dict]): A list of collected paths.

        """
        self.policy.log_diagnostics(paths)
        self.qf.log_diagnostics(paths)

    def process_samples(self, itr, paths):
        # pylint: disable=no-self-use
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            dict: Processed sample data, with keys
                * undiscounted_returns (list[float])
                * success_history (list[float])
                * complete (list[bool])

        """
        del itr

        success_history = [
            path['success_count'] / path['running_length'] for path in paths
        ]
        undiscounted_returns = [path['undiscounted_return'] for path in paths]

        # check if the last path is complete
        complete = [path['dones'][-1] for path in paths]

        samples_data = dict(undiscounted_returns=undiscounted_returns,
                            success_history=success_history,
                            complete=complete)

        return samples_data

    def init_opt(self):
        # pylint: disable=no-self-use
        """Initialize the optimization procedure.

        If using tensorflow, this may include declaring all the variables
        and compiling functions.

        """

    @abc.abstractmethod
    def optimize_policy(self, itr, samples_data):
        """Optimize policy network.

        Args:
            itr (int): Iterations.
            samples_data (list): Processed batch data.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """
        raise NotImplementedError

    def _obtain_evaluation_samples(self,
                                   env,
                                   num_trajs=100,
                                   max_path_length=1000):
        """Sample the policy for 10 trajectories and return average values.

        Args:
            env (garage.envs.GarageEnv): The environement used to obtain
                trajectories.
            num_trajs (int): Number of trajectories.
            max_path_length (int): Number of maximum steps in one batch.

        Returns:
            TrajectoryBatch: Evaluation trajectories, representing the best
                current performance of the algorithm.

        """
        paths = []

        for _ in range(num_trajs):
            path = rollout(env,
                           self.policy,
                           max_path_length=max_path_length,
                           deterministic=True)
            paths.append(path)
        return TrajectoryBatch.from_trajectory_list(self.env_spec, paths)
