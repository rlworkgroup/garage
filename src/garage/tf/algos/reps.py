"""Relative Entropy Policy Search implementation in Tensorflow."""
# yapf: disable
import collections

from dowel import logger, tabular
import numpy as np
import scipy.optimize
import tensorflow as tf

from garage import (_Default,
                    EpisodeBatch,
                    log_performance,
                    make_optimizer,
                    StepType)
from garage.np.algos import RLAlgorithm
from garage.sampler import RaySampler
from garage.tf import paths_to_tensors
from garage.tf.misc import tensor_utils
from garage.tf.misc.tensor_utils import flatten_inputs, graph_inputs
from garage.tf.optimizers import LbfgsOptimizer

# yapf: disable


# pylint: disable=differing-param-doc, differing-type-doc
class REPS(RLAlgorithm):  # noqa: D416
    """Relative Entropy Policy Search.

    References
    ----------
    [1] J. Peters, K. Mulling, and Y. Altun, "Relative Entropy Policy Search,"
        Artif. Intell., pp. 1607-1612, 2008.

    Example:
        $ python garage/examples/tf/reps_gym_cartpole.py

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_episode_length (int): Maximum length of a single episode.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        epsilon (float): Dual func parameter.
        l2_reg_dual (float): Coefficient for dual func l2 regularization.
        l2_reg_loss (float): Coefficient for policy loss l2 regularization.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): Arguments of the optimizer.
        dual_optimizer (object): Dual func optimizer.
        dual_optimizer_args (dict): Arguments of the dual optimizer.
        name (str): Name of the algorithm.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 max_episode_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 epsilon=0.5,
                 l2_reg_dual=0.,
                 l2_reg_loss=0.,
                 optimizer=LbfgsOptimizer,
                 optimizer_args=None,
                 dual_optimizer=scipy.optimize.fmin_l_bfgs_b,
                 dual_optimizer_args=None,
                 name='REPS'):
        optimizer_args = optimizer_args or dict(max_opt_itr=_Default(50))
        dual_optimizer_args = dual_optimizer_args or dict(maxiter=50)

        self.policy = policy
        self.max_episode_length = max_episode_length

        self._env_spec = env_spec
        self._baseline = baseline
        self._discount = discount
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._fixed_horizon = fixed_horizon

        self._name = name
        self._name_scope = tf.name_scope(self._name)
        self._old_policy = policy.clone('old_policy')

        self._feat_diff = None
        self._param_eta = None
        self._param_v = None
        self._f_dual = None
        self._f_dual_grad = None
        self._f_policy_kl = None
        self._policy_network = None
        self._old_policy_network = None

        self._optimizer = make_optimizer(optimizer, **optimizer_args)
        self._dual_optimizer = dual_optimizer
        self._dual_optimizer_args = dual_optimizer_args
        self._epsilon = float(epsilon)
        self._l2_reg_dual = float(l2_reg_dual)
        self._l2_reg_loss = float(l2_reg_loss)

        self._episode_reward_mean = collections.deque(maxlen=100)
        self.sampler_cls = RaySampler

        self.init_opt()

    def init_opt(self):
        """Initialize the optimization procedure."""
        pol_loss_inputs, pol_opt_inputs, dual_opt_inputs = self._build_inputs()
        self._policy_opt_inputs = pol_opt_inputs
        self._dual_opt_inputs = dual_opt_inputs

        pol_loss = self._build_policy_loss(pol_loss_inputs)
        self._optimizer.update_opt(loss=pol_loss,
                                   target=self.policy,
                                   inputs=flatten_inputs(
                                       self._policy_opt_inputs))

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): Experiment runner, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in runner.step_epochs():
            runner.step_path = runner.obtain_samples(runner.step_itr)
            last_return = self.train_once(runner.step_itr, runner.step_path)
            runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Average return.

        """
        # -- Stage: Calculate baseline
        paths = [
            dict(
                observations=path['observations'],
                actions=(
                    self._env_spec.action_space.flatten_n(  # noqa: E126
                        path['actions'])),
                rewards=path['rewards'],
                env_infos=path['env_infos'],
                agent_infos=path['agent_infos'],
                dones=np.array([
                    step_type == StepType.TERMINAL
                    for step_type in path['step_types']
                ])) for path in paths
        ]

        if hasattr(self._baseline, 'predict_n'):
            baseline_predictions = self._baseline.predict_n(paths)
        else:
            baseline_predictions = [
                self._baseline.predict(path) for path in paths
            ]

        # -- Stage: Pre-process samples based on collected paths
        samples_data = paths_to_tensors(paths, self.max_episode_length,
                                        baseline_predictions, self._discount,
                                        self._gae_lambda)

        # -- Stage: Run and calculate performance of the algorithm
        undiscounted_returns = log_performance(
            itr,
            EpisodeBatch.from_list(self._env_spec, paths),
            discount=self._discount)
        self._episode_reward_mean.extend(undiscounted_returns)
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self._episode_reward_mean))

        samples_data['average_return'] = np.mean(undiscounted_returns)

        logger.log('Optimizing policy...')
        self.optimize_policy(samples_data)
        return samples_data['average_return']

    def __getstate__(self):
        """Parameters to save in snapshot.

        Returns:
            dict: Parameters to save.

        """
        data = self.__dict__.copy()
        del data['_name_scope']
        del data['_policy_opt_inputs']
        del data['_dual_opt_inputs']
        del data['_f_dual']
        del data['_f_dual_grad']
        del data['_f_policy_kl']
        del data['_policy_network']
        del data['_old_policy_network']
        return data

    def __setstate__(self, state):
        """Parameters to restore from snapshot.

        Args:
            state (dict): Parameters to restore from.

        """
        self.__dict__ = state
        self._name_scope = tf.name_scope(self._name)
        self.init_opt()

    def optimize_policy(self, samples_data):
        """Optimize the policy using the samples.

        Args:
            samples_data (dict): Processed sample data.
                See garage.tf.paths_to_tensors() for details.

        """
        # Initial BFGS parameter values.
        x0 = np.hstack([self._param_eta, self._param_v])
        # Set parameter boundaries: \eta>=1e-12, v unrestricted.
        bounds = [(-np.inf, np.inf) for _ in x0]
        bounds[0] = (1e-12, np.inf)

        # Optimize dual
        eta_before = self._param_eta
        logger.log('Computing dual before')
        self._feat_diff = self._features(samples_data)
        dual_opt_input_values = self._dual_opt_input_values(samples_data)
        dual_before = self._f_dual(*dual_opt_input_values)
        logger.log('Optimizing dual')

        def eval_dual(x):
            """Evaluate dual function loss.

            Args:
                x (numpy.ndarray): Input to dual function.

            Returns:
                numpy.float64: Dual function loss.

            """
            self._param_eta = x[0]
            self._param_v = x[1:]
            dual_opt_input_values = self._dual_opt_input_values(samples_data)
            return self._f_dual(*dual_opt_input_values)

        def eval_dual_grad(x):
            """Evaluate gradient of dual function loss.

            Args:
                x (numpy.ndarray): Input to dual function.

            Returns:
                numpy.ndarray: Gradient of dual function loss.

            """
            self._param_eta = x[0]
            self._param_v = x[1:]
            dual_opt_input_values = self._dual_opt_input_values(samples_data)
            grad = self._f_dual_grad(*dual_opt_input_values)
            eta_grad = np.float(grad[0])
            v_grad = grad[1]
            return np.hstack([eta_grad, v_grad])

        params_ast, _, _ = self._dual_optimizer(func=eval_dual,
                                                x0=x0,
                                                fprime=eval_dual_grad,
                                                bounds=bounds,
                                                **self._dual_optimizer_args)

        logger.log('Computing dual after')
        self._param_eta, self._param_v = params_ast[0], params_ast[1:]
        dual_opt_input_values = self._dual_opt_input_values(samples_data)
        dual_after = self._f_dual(*dual_opt_input_values)

        # Optimize policy
        policy_opt_input_values = self._policy_opt_input_values(samples_data)
        logger.log('Computing policy loss before')
        loss_before = self._optimizer.loss(policy_opt_input_values)
        logger.log('Computing policy KL before')
        policy_kl_before = self._f_policy_kl(*policy_opt_input_values)
        logger.log('Optimizing policy')
        self._optimizer.optimize(policy_opt_input_values)
        logger.log('Computing policy KL')
        policy_kl = self._f_policy_kl(*policy_opt_input_values)
        logger.log('Computing policy loss after')
        loss_after = self._optimizer.loss(policy_opt_input_values)
        tabular.record('EtaBefore', eta_before)
        tabular.record('EtaAfter', self._param_eta)
        tabular.record('DualBefore', dual_before)
        tabular.record('DualAfter', dual_after)
        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)
        self._old_policy.parameters = self.policy.parameters

    def _build_inputs(self):
        """Build input variables.

        Returns:
            namedtuple: Collection of variables to compute policy loss.
            namedtuple: Collection of variables to do policy optimization.

        """
        observation_space = self.policy.observation_space
        action_space = self.policy.action_space

        with tf.name_scope('inputs'):
            obs_var = observation_space.to_tf_placeholder(
                name='obs',
                batch_dims=2)  # yapf: disable
            action_var = action_space.to_tf_placeholder(
                name='action',
                batch_dims=2)  # yapf: disable
            reward_var = tensor_utils.new_tensor(
                name='reward',
                ndim=2,
                dtype=tf.float32)  # yapf: disable
            valid_var = tensor_utils.new_tensor(
                name='valid',
                ndim=2,
                dtype=tf.float32)  # yapf: disable
            feat_diff = tensor_utils.new_tensor(
                name='feat_diff',
                ndim=2,
                dtype=tf.float32)  # yapf: disable
            param_v = tensor_utils.new_tensor(
                name='param_v',
                ndim=1,
                dtype=tf.float32)  # yapf: disable
            param_eta = tensor_utils.new_tensor(
                name='param_eta',
                ndim=0,
                dtype=tf.float32)  # yapf: disable
            policy_state_info_vars = {
                k: tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None] * 2 + list(shape),
                    name=k)
                for k, shape in self.policy.state_info_specs
            }  # yapf: disable
            policy_state_info_vars_list = [
                policy_state_info_vars[k]
                for k in self.policy.state_info_keys
            ]  # yapf: disable

        self._policy_network = self.policy.build(obs_var, name='policy')
        self._old_policy_network = self._old_policy.build(obs_var,
                                                          name='policy')

        policy_loss_inputs = graph_inputs(
            'PolicyLossInputs',
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            valid_var=valid_var,
            feat_diff=feat_diff,
            param_eta=param_eta,
            param_v=param_v,
            policy_state_info_vars=policy_state_info_vars,
        )
        policy_opt_inputs = graph_inputs(
            'PolicyOptInputs',
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            valid_var=valid_var,
            feat_diff=feat_diff,
            param_eta=param_eta,
            param_v=param_v,
            policy_state_info_vars_list=policy_state_info_vars_list,
        )
        dual_opt_inputs = graph_inputs(
            'DualOptInputs',
            reward_var=reward_var,
            valid_var=valid_var,
            feat_diff=feat_diff,
            param_eta=param_eta,
            param_v=param_v,
            policy_state_info_vars_list=policy_state_info_vars_list,
        )

        return policy_loss_inputs, policy_opt_inputs, dual_opt_inputs

    def _build_policy_loss(self, i):
        """Build policy loss and other output tensors.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy loss.
            tf.Tensor: Mean policy KL divergence.

        Raises:
            NotImplementedError: If is_recurrent is True.

        """
        pol_dist = self._policy_network.dist
        old_pol_dist = self._old_policy_network.dist

        # Initialize dual params
        self._param_eta = 15.
        self._param_v = np.random.rand(
            self._env_spec.observation_space.flat_dim * 2 + 4)

        with tf.name_scope('bellman_error'):
            delta_v = tf.boolean_mask(i.reward_var,
                                      i.valid_var) + tf.tensordot(
                                          i.feat_diff, i.param_v, 1)

        with tf.name_scope('policy_loss'):
            ll = pol_dist.log_prob(i.action_var)
            ll = tf.boolean_mask(ll, i.valid_var)
            loss = -tf.reduce_mean(
                ll * tf.exp(delta_v / i.param_eta -
                            tf.reduce_max(delta_v / i.param_eta)))

            reg_params = self.policy.get_regularizable_vars()
            loss += self._l2_reg_loss * tf.reduce_sum(
                [tf.reduce_mean(tf.square(param))
                 for param in reg_params]) / len(reg_params)

        with tf.name_scope('kl'):
            kl = old_pol_dist.kl_divergence(pol_dist)
            pol_mean_kl = tf.reduce_mean(kl)

        with tf.name_scope('dual'):
            dual_loss = i.param_eta * self._epsilon + (
                i.param_eta * tf.math.log(
                    tf.reduce_mean(
                        tf.exp(delta_v / i.param_eta -
                               tf.reduce_max(delta_v / i.param_eta)))) +
                i.param_eta * tf.reduce_max(delta_v / i.param_eta))

            dual_loss += self._l2_reg_dual * (tf.square(i.param_eta) +
                                              tf.square(1 / i.param_eta))

            dual_grad = tf.gradients(dual_loss, [i.param_eta, i.param_v])

        # yapf: disable
        self._f_dual = tensor_utils.compile_function(
            flatten_inputs(self._dual_opt_inputs),
            dual_loss,
            log_name='f_dual')
        # yapf: enable

        self._f_dual_grad = tensor_utils.compile_function(
            flatten_inputs(self._dual_opt_inputs),
            dual_grad,
            log_name='f_dual_grad')

        self._f_policy_kl = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs),
            pol_mean_kl,
            log_name='f_policy_kl')

        return loss

    def _dual_opt_input_values(self, samples_data):
        """Update dual func optimize input values based on samples data.

        Args:
            samples_data (dict): Processed sample data.
                See garage.tf.paths_to_tensors() for details.

        Returns:
            list(np.ndarray): Flatten dual function optimization input values.

        """
        policy_state_info_list = [
            samples_data['agent_infos'][k]
            for k in self.policy.state_info_keys
        ]  # yapf: disable

        # pylint: disable=unexpected-keyword-arg
        dual_opt_input_values = self._dual_opt_inputs._replace(
            reward_var=samples_data['rewards'],
            valid_var=samples_data['valids'],
            feat_diff=self._feat_diff,
            param_eta=self._param_eta,
            param_v=self._param_v,
            policy_state_info_vars_list=policy_state_info_list,
        )

        return flatten_inputs(dual_opt_input_values)

    def _policy_opt_input_values(self, samples_data):
        """Update policy optimize input values based on samples data.

        Args:
            samples_data (dict): Processed sample data.
                See garage.tf.paths_to_tensors() for details.

        Returns:
            list(np.ndarray): Flatten policy optimization input values.

        """
        policy_state_info_list = [
            samples_data['agent_infos'][k]
            for k in self.policy.state_info_keys
        ]  # yapf: disable

        # pylint: disable=unexpected-keyword-arg
        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=samples_data['observations'],
            action_var=samples_data['actions'],
            reward_var=samples_data['rewards'],
            valid_var=samples_data['valids'],
            feat_diff=self._feat_diff,
            param_eta=self._param_eta,
            param_v=self._param_v,
            policy_state_info_vars_list=policy_state_info_list,
        )

        return flatten_inputs(policy_opt_input_values)

    def _features(self, samples_data):
        """Get valid view features based on samples data.

        Args:
            samples_data (dict): Processed sample data.
                See garage.tf.paths_to_tensors() for details.

        Returns:
            numpy.ndarray: Features for training.

        """
        paths = samples_data['paths']
        feat_diff = []
        for path in paths:
            o = np.clip(path['observations'],
                        self._env_spec.observation_space.low,
                        self._env_spec.observation_space.high)
            lr = len(path['rewards'])
            al = np.arange(lr).reshape(-1, 1) / self.max_episode_length
            feats = np.concatenate(
                [o, o**2, al, al**2, al**3,
                 np.ones((lr, 1))], axis=1)
            # pylint: disable=unsubscriptable-object
            feats = np.vstack([feats, np.zeros(feats.shape[1])])
            feat_diff.append(feats[1:] - feats[:-1])

        return np.vstack(feat_diff)
