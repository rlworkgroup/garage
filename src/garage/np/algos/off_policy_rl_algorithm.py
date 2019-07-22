from garage.np.algos import RLAlgorithm


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
        n_epoch_cycles (int): Number of train_once calls per epoch.
        max_path_length (int): Maximum path length. The episode will
            terminate when length of trajectory reaches max_path_length.
        n_train_steps (int): Training steps.
        buffer_batch_size (int): Batch size for replay buffer.
        min_buffer_size (int): The minimum buffer size for replay buffer.
        rollout_batch_size (int): Roll out batch size.
        reward_scale (float): Reward scale.
        input_include_goal (bool): Whether input includes goal.
        smooth_return (bool): Whether to smooth the return.
        exploration_strategy (garage.np.exploration_strategies.
            ExplorationStrategy): Exploration strategy.

    """
    def __init__(self,
                 env_spec,
                 policy,
                 qf,
                 replay_buffer,
                 use_target=False,
                 discount=0.99,
                 n_epoch_cycles=20,
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
        self.n_epoch_cycles = n_epoch_cycles
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
        self.init_opt()

    def train(self, runner, batch_size):
        last_return = None

        for epoch in runner.step_epochs():
            for cycle in range(self.n_epoch_cycles):
                runner.step_path = runner.obtain_samples(
                    runner.step_itr, batch_size)
                last_return = self.train_once(runner.step_itr,
                                              runner.step_path)
                runner.step_itr += 1

        return last_return

    def log_diagnostics(self, paths):
        """Log diagnostic information on current paths."""
        self.policy.log_diagnostics(paths)
        self.qf.log_diagnostics(paths)

    def process_samples(self, itr, paths):
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
        """
        Initialize the optimization procedure.

        If using tensorflow, this may
        include declaring all the variables and compiling functions.
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """Return data saved in the snapshot for this iteration."""
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        """Optimize policy network."""
        raise NotImplementedError
