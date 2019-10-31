"""Relative Entropy Policy Search implementation in Tensorflow."""
from dowel import logger, tabular
import numpy as np
import scipy.optimize
import tensorflow as tf

from garage.tf.algos.batch_polopt import BatchPolopt
from garage.tf.misc import tensor_utils
from garage.tf.misc.tensor_utils import filter_valids
from garage.tf.misc.tensor_utils import filter_valids_dict
from garage.tf.misc.tensor_utils import flatten_batch
from garage.tf.misc.tensor_utils import flatten_batch_dict
from garage.tf.misc.tensor_utils import flatten_inputs
from garage.tf.misc.tensor_utils import graph_inputs
from garage.tf.optimizers import LbfgsOptimizer


class REPS(BatchPolopt):
    """Relative Entropy Policy Search.

    References
    ----------
    [1] J. Peters, K. Mulling, and Y. Altun, "Relative Entropy Policy Search,"
        Artif. Intell., pp. 1607-1612, 2008.

    Example:
        $ python garage/examples/tf/reps_gym_cartpole.py

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_path_length (int): Maximum length of a single rollout.
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
        epsilon (float): dual func parameter.
        l2_reg_dual (float): coefficient for dual func l2 regularization.
        l2_reg_loss (float): coefficient for policy loss l2 regularization.
        dual_optimizer (object): dual func optimizer.
        dual_optimizer_args (dict): arguments of the dual optimizer.
        name (str): Name of the algorithm.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 epsilon=0.5,
                 l2_reg_dual=0.,
                 l2_reg_loss=0.,
                 optimizer=LbfgsOptimizer,
                 optimizer_args=dict(max_opt_itr=50),
                 dual_optimizer=scipy.optimize.fmin_l_bfgs_b,
                 dual_optimizer_args=dict(maxiter=50),
                 name='REPS'):
        self.name = name
        self._name_scope = tf.name_scope(self.name)

        with self._name_scope:
            self.optimizer = optimizer(**optimizer_args)
            self.dual_optimizer = dual_optimizer
            self.dual_optimizer_args = dual_optimizer_args
            self.epsilon = float(epsilon)
            self.l2_reg_dual = float(l2_reg_dual)
            self.l2_reg_loss = float(l2_reg_loss)

        super(REPS, self).__init__(env_spec=env_spec,
                                   policy=policy,
                                   baseline=baseline,
                                   max_path_length=max_path_length,
                                   discount=discount,
                                   gae_lambda=gae_lambda,
                                   center_adv=center_adv,
                                   positive_adv=positive_adv,
                                   fixed_horizon=fixed_horizon)

    def init_opt(self):
        """Initialize the optimization procedure."""
        pol_loss_inputs, pol_opt_inputs, dual_opt_inputs = self._build_inputs()
        self._policy_opt_inputs = pol_opt_inputs
        self._dual_opt_inputs = dual_opt_inputs

        pol_loss = self._build_policy_loss(pol_loss_inputs)
        self.optimizer.update_opt(loss=pol_loss,
                                  target=self.policy,
                                  inputs=flatten_inputs(
                                      self._policy_opt_inputs))

    def __getstate__(self):
        """Object.__getstate__."""
        data = self.__dict__.copy()
        del data['_name_scope']
        del data['_policy_opt_inputs']
        del data['_dual_opt_inputs']
        del data['f_dual']
        del data['f_dual_grad']
        del data['f_policy_kl']
        return data

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__ = state
        self._name_scope = tf.name_scope(self.name)
        self.init_opt()

    def get_itr_snapshot(self, itr):
        """Return the data should saved in the snapshot."""
        return dict(
            itr=itr,
            policy=self.policy,
        )

    def optimize_policy(self, itr, samples_data):
        """Perform the policy optimization."""
        # Initial BFGS parameter values.
        x0 = np.hstack([self.param_eta, self.param_v])
        # Set parameter boundaries: \eta>=1e-12, v unrestricted.
        bounds = [(-np.inf, np.inf) for _ in x0]
        bounds[0] = (1e-12, np.inf)

        # Optimize dual
        eta_before = self.param_eta
        logger.log('Computing dual before')
        self.feat_diff = self._features(samples_data)
        dual_opt_input_values = self._dual_opt_input_values(samples_data)
        dual_before = self.f_dual(*dual_opt_input_values)
        logger.log('Optimizing dual')

        def eval_dual(x):
            self.param_eta = x[0]
            self.param_v = x[1:]
            dual_opt_input_values = self._dual_opt_input_values(samples_data)
            return self.f_dual(*dual_opt_input_values)

        def eval_dual_grad(x):
            self.param_eta = x[0]
            self.param_v = x[1:]
            dual_opt_input_values = self._dual_opt_input_values(samples_data)
            grad = self.f_dual_grad(*dual_opt_input_values)
            eta_grad = np.float(grad[0])
            v_grad = grad[1]
            return np.hstack([eta_grad, v_grad])

        params_ast, _, _ = self.dual_optimizer(func=eval_dual,
                                               x0=x0,
                                               fprime=eval_dual_grad,
                                               bounds=bounds,
                                               **self.dual_optimizer_args)

        logger.log('Computing dual after')
        self.param_eta, self.param_v = params_ast[0], params_ast[1:]
        dual_opt_input_values = self._dual_opt_input_values(samples_data)
        dual_after = self.f_dual(*dual_opt_input_values)

        # Optimize policy
        policy_opt_input_values = self._policy_opt_input_values(samples_data)
        logger.log('Computing policy loss before')
        loss_before = self.optimizer.loss(policy_opt_input_values)
        logger.log('Computing policy KL before')
        policy_kl_before = self.f_policy_kl(*policy_opt_input_values)
        logger.log('Optimizing policy')
        self.optimizer.optimize(policy_opt_input_values)
        logger.log('Computing policy KL')
        policy_kl = self.f_policy_kl(*policy_opt_input_values)
        logger.log('Computing policy loss after')
        loss_after = self.optimizer.loss(policy_opt_input_values)
        tabular.record('EtaBefore', eta_before)
        tabular.record('EtaAfter', self.param_eta)
        tabular.record('DualBefore', dual_before)
        tabular.record('DualAfter', dual_after)
        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)

    def _build_inputs(self):
        """Decalre graph inputs variables."""
        observation_space = self.policy.observation_space
        action_space = self.policy.action_space
        policy_dist = self.policy.distribution

        with tf.name_scope('inputs'):
            obs_var = observation_space.to_tf_placeholder(
                name='obs',
                batch_dims=2)   # yapf: disable
            action_var = action_space.to_tf_placeholder(
                name='action',
                batch_dims=2)   # yapf: disable
            reward_var = tensor_utils.new_tensor(
                name='reward',
                ndim=2,
                dtype=tf.float32)   # yapf: disable
            valid_var = tensor_utils.new_tensor(
                name='valid',
                ndim=2,
                dtype=tf.float32)   # yapf: disable
            feat_diff = tensor_utils.new_tensor(
                name='feat_diff',
                ndim=2,
                dtype=tf.float32)   # yapf: disable
            param_v = tensor_utils.new_tensor(
                name='param_v',
                ndim=1,
                dtype=tf.float32)   # yapf: disable
            param_eta = tensor_utils.new_tensor(
                name='param_eta',
                ndim=0,
                dtype=tf.float32)   # yapf: disable
            policy_state_info_vars = {
                k: tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None] * 2 + list(shape),
                    name=k)
                for k, shape in self.policy.state_info_specs
            }   # yapf: disable
            policy_state_info_vars_list = [
                policy_state_info_vars[k]
                for k in self.policy.state_info_keys
            ]   # yapf: disable

            policy_old_dist_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name='policy_old_%s' % k)
                for k, shape in policy_dist.dist_info_specs
            }
            policy_old_dist_info_vars_list = [
                policy_old_dist_info_vars[k]
                for k in policy_dist.dist_info_keys
            ]

            with tf.name_scope('flat'):
                obs_flat = flatten_batch(obs_var, name='obs_flat')
                action_flat = flatten_batch(action_var, name='action_flat')
                reward_flat = flatten_batch(reward_var, name='reward_flat')
                valid_flat = flatten_batch(valid_var, name='valid_flat')
                feat_diff_flat = flatten_batch(
                    feat_diff,
                    name='feat_diff_flat')  # yapf: disable
                policy_state_info_vars_flat = flatten_batch_dict(
                    policy_state_info_vars,
                    name='policy_state_info_vars_flat')  # yapf: disable
                policy_old_dist_info_vars_flat = flatten_batch_dict(
                    policy_old_dist_info_vars,
                    name='policy_old_dist_info_vars_flat')

            with tf.name_scope('valid'):
                reward_valid = filter_valids(
                    reward_flat,
                    valid_flat,
                    name='reward_valid')   # yapf: disable
                action_valid = filter_valids(
                    action_flat,
                    valid_flat,
                    name='action_valid')    # yapf: disable
                policy_state_info_vars_valid = filter_valids_dict(
                    policy_state_info_vars_flat,
                    valid_flat,
                    name='policy_state_info_vars_valid')
                policy_old_dist_info_vars_valid = filter_valids_dict(
                    policy_old_dist_info_vars_flat,
                    valid_flat,
                    name='policy_old_dist_info_vars_valid')

        pol_flat = graph_inputs(
            'PolicyLossInputsFlat',
            obs_var=obs_flat,
            action_var=action_flat,
            reward_var=reward_flat,
            valid_var=valid_flat,
            feat_diff=feat_diff_flat,
            policy_state_info_vars=policy_state_info_vars_flat,
            policy_old_dist_info_vars=policy_old_dist_info_vars_flat,
        )
        pol_valid = graph_inputs(
            'PolicyLossInputsValid',
            reward_var=reward_valid,
            action_var=action_valid,
            policy_state_info_vars=policy_state_info_vars_valid,
            policy_old_dist_info_vars=policy_old_dist_info_vars_valid,
        )
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
            policy_old_dist_info_vars=policy_old_dist_info_vars,
            flat=pol_flat,
            valid=pol_valid,
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
            policy_old_dist_info_vars_list=policy_old_dist_info_vars_list,
        )
        dual_opt_inputs = graph_inputs(
            'DualOptInputs',
            reward_var=reward_var,
            valid_var=valid_var,
            feat_diff=feat_diff,
            param_eta=param_eta,
            param_v=param_v,
            policy_state_info_vars_list=policy_state_info_vars_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_vars_list,
        )

        return policy_loss_inputs, policy_opt_inputs, dual_opt_inputs

    def _build_policy_loss(self, i):
        """Initialize policy loss complie function based on inputs i."""
        pol_dist = self.policy.distribution
        is_recurrent = self.policy.recurrent

        # Initialize dual params
        self.param_eta = 15.
        self.param_v = np.random.rand(
            self.env_spec.observation_space.flat_dim * 2 + 4)

        if is_recurrent:
            raise NotImplementedError

        policy_dist_info_flat = self.policy.dist_info_sym(
            i.flat.obs_var,
            i.flat.policy_state_info_vars,
            name='policy_dist_info_flat')

        policy_dist_info_valid = filter_valids_dict(
            policy_dist_info_flat,
            i.flat.valid_var,
            name='policy_dist_info_valid')

        with tf.name_scope('bellman_error'):
            delta_v = i.valid.reward_var + tf.tensordot(
                i.feat_diff, i.param_v, 1)

        with tf.name_scope('policy_loss'):
            ll = pol_dist.log_likelihood_sym(i.valid.action_var,
                                             policy_dist_info_valid)
            loss = -tf.reduce_mean(
                ll * tf.exp(delta_v / i.param_eta -
                            tf.reduce_max(delta_v / i.param_eta)))

            reg_params = self.policy.get_regularizable_vars()
            loss += self.l2_reg_loss * tf.reduce_sum(
                [tf.reduce_mean(tf.square(param))
                 for param in reg_params]) / len(reg_params)

        with tf.name_scope('kl'):
            kl = pol_dist.kl_sym(
                i.valid.policy_old_dist_info_vars,
                policy_dist_info_valid,
            )
            pol_mean_kl = tf.reduce_mean(kl)

        with tf.name_scope('dual'):
            dual_loss = i.param_eta * self.epsilon + i.param_eta * tf.math.log(
                tf.reduce_mean(
                    tf.exp(delta_v / i.param_eta -
                           tf.reduce_max(delta_v / i.param_eta)))
            ) + i.param_eta * tf.reduce_max(delta_v / i.param_eta)

            dual_loss += self.l2_reg_dual * (tf.square(i.param_eta) +
                                             tf.square(1 / i.param_eta))

            dual_grad = tf.gradients(dual_loss, [i.param_eta, i.param_v])

        # yapf: disable
        self.f_dual = tensor_utils.compile_function(
            flatten_inputs(self._dual_opt_inputs),
            dual_loss,
            log_name='f_dual')
        # yapf: enable

        self.f_dual_grad = tensor_utils.compile_function(
            flatten_inputs(self._dual_opt_inputs),
            dual_grad,
            log_name='f_dual_grad')

        self.f_policy_kl = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs),
            pol_mean_kl,
            log_name='f_policy_kl')

        return loss

    def _dual_opt_input_values(self, samples_data):
        """Update dual func optimize input values based on samples data."""
        policy_state_info_list = [
            samples_data['agent_infos'][k]
            for k in self.policy.state_info_keys
        ]   # yapf: disable
        policy_old_dist_info_list = [
            samples_data['agent_infos'][k]
            for k in self.policy.distribution.dist_info_keys
        ]

        dual_opt_input_values = self._dual_opt_inputs._replace(
            reward_var=samples_data['rewards'],
            valid_var=samples_data['valids'],
            feat_diff=self.feat_diff,
            param_eta=self.param_eta,
            param_v=self.param_v,
            policy_state_info_vars_list=policy_state_info_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_list,
        )

        return flatten_inputs(dual_opt_input_values)

    def _policy_opt_input_values(self, samples_data):
        """Update policy optimize input values based on samples data."""
        policy_state_info_list = [
            samples_data['agent_infos'][k]
            for k in self.policy.state_info_keys
        ]   # yapf: disable
        policy_old_dist_info_list = [
            samples_data['agent_infos'][k]
            for k in self.policy.distribution.dist_info_keys
        ]

        # pylint: disable=locally-disabled, unexpected-keyword-arg
        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=samples_data['observations'],
            action_var=samples_data['actions'],
            reward_var=samples_data['rewards'],
            valid_var=samples_data['valids'],
            feat_diff=self.feat_diff,
            param_eta=self.param_eta,
            param_v=self.param_v,
            policy_state_info_vars_list=policy_state_info_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_list,
        )

        return flatten_inputs(policy_opt_input_values)

    def _features(self, samples_data):
        """Get valid view features based on samples data."""
        paths = samples_data['paths']
        feat_diff = []
        for path in paths:
            o = np.clip(path['observations'],
                        self.env_spec.observation_space.low,
                        self.env_spec.observation_space.high)
            lr = len(path['rewards'])
            al = np.arange(lr).reshape(-1, 1) / self.max_path_length
            feats = np.concatenate(
                [o, o**2, al, al**2, al**3,
                 np.ones((lr, 1))], axis=1)
            feats = np.vstack([feats, np.zeros(feats.shape[1])])
            feat_diff.append(feats[1:] - feats[:-1])

        return np.vstack(feat_diff)
