"""
This module implements a class for off-policy rl algorithms.

Off-policy algorithms such as DQN, DDPG can inherit from it.
"""
from garage.algos import RLAlgorithm
from garage.tf.plotter import Plotter
from garage.tf.samplers import BatchSampler
from garage.tf.samplers import OffPolicyVectorizedSampler


class OffPolicyRLAlgorithm(RLAlgorithm):
    """This class implements OffPolicyRLAlgorithm."""

    def __init__(
            self,
            env,
            policy,
            qf,
            replay_buffer,
            use_target=False,
            discount=0.99,
            n_epochs=500,
            n_epoch_cycles=20,
            max_path_length=100,
            n_train_steps=50,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            rollout_batch_size=1,
            reward_scale=1.,
            input_include_goal=False,
            smooth_return=True,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            plot=False,
            pause_for_plot=False,
            exploration_strategy=None,
    ):
        """Construct an OffPolicyRLAlgorithm class."""
        self.env = env
        self.policy = policy
        self.qf = qf
        self.replay_buffer = replay_buffer
        self.n_epochs = n_epochs
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
        if sampler_cls is None:
            if policy.vectorized and not force_batch_sampler:
                sampler_cls = OffPolicyVectorizedSampler
            else:
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.max_path_length = max_path_length
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.es = exploration_strategy
        self.init_opt()

    def start_worker(self, sess):
        """Initialize sampler and plotter."""
        self.sampler.start_worker()
        if self.plot:
            self.plotter = Plotter(self.env, self.policy, sess)
            self.plotter.start()

    def shutdown_worker(self):
        """Close sampler and plotter."""
        self.sampler.shutdown_worker()
        if self.plot:
            self.plotter.close()

    def obtain_samples(self, itr):
        """Sample data for this iteration."""
        return self.sampler.obtain_samples(itr)

    def process_samples(self, itr, paths):
        """Process samples from rollout paths."""
        return self.sampler.process_samples(itr, paths)

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
