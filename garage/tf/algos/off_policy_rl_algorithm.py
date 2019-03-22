"""
This module implements a class for off-policy rl algorithms.

Off-policy algorithms such as DQN, DDPG can inherit from it.
"""
from garage.algos import RLAlgorithm


class OffPolicyRLAlgorithm(RLAlgorithm):
    """This class implements OffPolicyRLAlgorithm."""

    def __init__(
            self,
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
            exploration_strategy=None,
    ):
        """Construct an OffPolicyRLAlgorithm class."""
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

    def log_diagnostics(self, paths):
        """Log diagnostic information on current paths."""
        self.policy.log_diagnostics(paths)
        self.qf.log_diagnostics(paths)

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
