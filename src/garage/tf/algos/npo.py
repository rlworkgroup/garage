"""Natural Policy Gradient Optimization."""
from dowel import logger, tabular
import numpy as np
import tensorflow as tf

from garage.misc import tensor_utils as np_tensor_utils
from garage.tf.algos.batch_polopt import BatchPolopt
from garage.tf.misc.tensor_utils import center_advs
from garage.tf.misc.tensor_utils import compile_function
from garage.tf.misc.tensor_utils import compute_advantages
from garage.tf.misc.tensor_utils import concat_tensor_list
from garage.tf.misc.tensor_utils import discounted_returns
from garage.tf.misc.tensor_utils import filter_valids
from garage.tf.misc.tensor_utils import filter_valids_dict
from garage.tf.misc.tensor_utils import flatten_batch
from garage.tf.misc.tensor_utils import flatten_batch_dict
from garage.tf.misc.tensor_utils import flatten_inputs
from garage.tf.misc.tensor_utils import graph_inputs
from garage.tf.misc.tensor_utils import new_tensor
from garage.tf.misc.tensor_utils import positive_advs
from garage.tf.optimizers import LbfgsOptimizer


class NPO(BatchPolopt):
    """
    Natural Policy Gradient Optimization.

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
        pg_loss (str): A string from: 'vanilla', 'surrogate',
            'surrogate_clip'. The type of loss functions to use.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies, as in PPO.
        max_kl_step (float): The maximum KL divergence between old and new
            policies, as in TRPO.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        flatten_input (bool): Whether to flatten input along the observation
            dimension. If True, for example, an observation with shape (2, 4)
            will be flattened to 8.
        name (str): The name of the algorithm.
    Note:
        sane defaults for entropy configuration:
            - entropy_method='max', center_adv=False, stop_gradient=True
              (center_adv normalizes the advantages tensor, which will
              significantly alleviate the effect of entropy. It is also
              recommended to turn off entropy gradient so that the agent
              will focus on high-entropy actions instead of increasing the
              variance of the distribution.)
            - entropy_method='regularized', stop_gradient=False,
              use_neg_logli_entropy=False
    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='surrogate',
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 flatten_input=True,
                 name='NPO'):
        self.name = name
        self._name_scope = tf.name_scope(self.name)
        self._use_softplus_entropy = use_softplus_entropy
        self._use_neg_logli_entropy = use_neg_logli_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._pg_loss = pg_loss
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = LbfgsOptimizer

        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          use_neg_logli_entropy,
                                          policy_ent_coeff)

        if pg_loss not in ['vanilla', 'surrogate', 'surrogate_clip']:
            raise ValueError('Invalid pg_loss')

        with self._name_scope:
            self.optimizer = optimizer(**optimizer_args)
            self.lr_clip_range = float(lr_clip_range)
            self.max_kl_step = float(max_kl_step)
            self.policy_ent_coeff = float(policy_ent_coeff)

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         scope=scope,
                         max_path_length=max_path_length,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         fixed_horizon=fixed_horizon,
                         flatten_input=flatten_input)

    def init_opt(self):
        """Initialize optimizater."""
        pol_loss_inputs, pol_opt_inputs = self._build_inputs()
        self._policy_opt_inputs = pol_opt_inputs

        pol_loss, pol_kl = self._build_policy_loss(pol_loss_inputs)
        self.optimizer.update_opt(loss=pol_loss,
                                  target=self.policy,
                                  leq_constraint=(pol_kl, self.max_kl_step),
                                  inputs=flatten_inputs(
                                      self._policy_opt_inputs),
                                  constraint_name='mean_kl')

        return dict()

    def optimize_policy(self, itr, samples_data):
        """Optimize policy."""
        policy_opt_input_values = self._policy_opt_input_values(samples_data)
        # Train policy network
        logger.log('Computing loss before')
        loss_before = self.optimizer.loss(policy_opt_input_values)
        logger.log('Computing KL before')
        policy_kl_before = self.f_policy_kl(*policy_opt_input_values)
        logger.log('Optimizing')
        self.optimizer.optimize(policy_opt_input_values)
        logger.log('Computing KL after')
        policy_kl = self.f_policy_kl(*policy_opt_input_values)
        logger.log('Computing loss after')
        loss_after = self.optimizer.loss(policy_opt_input_values)
        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)
        pol_ent = self.f_policy_entropy(*policy_opt_input_values)
        tabular.record('{}/Entropy'.format(self.policy.name), np.mean(pol_ent))

        self._fit_baseline(samples_data)

    def get_itr_snapshot(self, itr):
        """Get iteration snapshot."""
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
        )

    def _build_inputs(self):
        observation_space = self.policy.observation_space
        action_space = self.policy.action_space

        policy_dist = self.policy.distribution

        with tf.name_scope('inputs'):
            if self.flatten_input:
                obs_var = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, None, observation_space.flat_dim],
                    name='obs')
            else:
                obs_var = observation_space.to_tf_placeholder(name='obs',
                                                              batch_dims=2)
            action_var = action_space.to_tf_placeholder(name='action',
                                                        batch_dims=2)
            reward_var = new_tensor(name='reward', ndim=2, dtype=tf.float32)
            valid_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=[None, None],
                                                 name='valid')
            baseline_var = new_tensor(name='baseline',
                                      ndim=2,
                                      dtype=tf.float32)

            policy_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name=k)
                for k, shape in self.policy.state_info_specs
            }
            policy_state_info_vars_list = [
                policy_state_info_vars[k] for k in self.policy.state_info_keys
            ]

            # old policy distribution
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

            # flattened view
            with tf.name_scope('flat'):
                obs_flat = flatten_batch(obs_var, name='obs_flat')
                action_flat = flatten_batch(action_var, name='action_flat')
                reward_flat = flatten_batch(reward_var, name='reward_flat')
                valid_flat = flatten_batch(valid_var, name='valid_flat')
                policy_state_info_vars_flat = flatten_batch_dict(
                    policy_state_info_vars, name='policy_state_info_vars_flat')
                policy_old_dist_info_vars_flat = flatten_batch_dict(
                    policy_old_dist_info_vars,
                    name='policy_old_dist_info_vars_flat')

            # valid view
            with tf.name_scope('valid'):
                action_valid = filter_valids(action_flat,
                                             valid_flat,
                                             name='action_valid')
                policy_state_info_vars_valid = filter_valids_dict(
                    policy_state_info_vars_flat,
                    valid_flat,
                    name='policy_state_info_vars_valid')
                policy_old_dist_info_vars_valid = filter_valids_dict(
                    policy_old_dist_info_vars_flat,
                    valid_flat,
                    name='policy_old_dist_info_vars_valid')

        # policy loss and optimizer inputs
        pol_flat = graph_inputs(
            'PolicyLossInputsFlat',
            obs_var=obs_flat,
            action_var=action_flat,
            reward_var=reward_flat,
            valid_var=valid_flat,
            policy_state_info_vars=policy_state_info_vars_flat,
            policy_old_dist_info_vars=policy_old_dist_info_vars_flat,
        )
        pol_valid = graph_inputs(
            'PolicyLossInputsValid',
            action_var=action_valid,
            policy_state_info_vars=policy_state_info_vars_valid,
            policy_old_dist_info_vars=policy_old_dist_info_vars_valid,
        )
        policy_loss_inputs = graph_inputs(
            'PolicyLossInputs',
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            valid_var=valid_var,
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
            baseline_var=baseline_var,
            valid_var=valid_var,
            policy_state_info_vars_list=policy_state_info_vars_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_vars_list,
        )

        return policy_loss_inputs, policy_opt_inputs

    def _build_policy_loss(self, i):
        pol_dist = self.policy.distribution
        policy_entropy = self._build_entropy_term(i)
        rewards = i.reward_var

        if self._maximum_entropy:
            with tf.name_scope('augmented_rewards'):
                rewards = i.reward_var + self.policy_ent_coeff * policy_entropy

        with tf.name_scope('policy_loss'):
            adv = compute_advantages(self.discount,
                                     self.gae_lambda,
                                     self.max_path_length,
                                     i.baseline_var,
                                     rewards,
                                     name='adv')

            adv_flat = flatten_batch(adv, name='adv_flat')
            adv_valid = filter_valids(adv_flat,
                                      i.flat.valid_var,
                                      name='adv_valid')

            if self.policy.recurrent:
                adv = tf.reshape(adv, [-1, self.max_path_length])

            # Optionally normalize advantages
            eps = tf.constant(1e-8, dtype=tf.float32)
            if self.center_adv:
                if self.policy.recurrent:
                    adv = center_advs(adv, axes=[0], eps=eps)
                else:
                    adv_valid = center_advs(adv_valid, axes=[0], eps=eps)

            if self.positive_adv:
                if self.policy.recurrent:
                    adv = positive_advs(adv, eps)
                else:
                    adv_valid = positive_advs(adv_valid, eps)

            if self.policy.recurrent:
                policy_dist_info = self.policy.dist_info_sym(
                    i.obs_var,
                    i.policy_state_info_vars,
                    name='policy_dist_info')
            else:
                policy_dist_info_flat = self.policy.dist_info_sym(
                    i.flat.obs_var,
                    i.flat.policy_state_info_vars,
                    name='policy_dist_info_flat')

                policy_dist_info_valid = filter_valids_dict(
                    policy_dist_info_flat,
                    i.flat.valid_var,
                    name='policy_dist_info_valid')

                policy_dist_info = policy_dist_info_valid

            # Calculate loss function and KL divergence
            with tf.name_scope('kl'):
                if self.policy.recurrent:
                    kl = pol_dist.kl_sym(
                        i.policy_old_dist_info_vars,
                        policy_dist_info,
                    )
                    pol_mean_kl = tf.reduce_sum(
                        kl * i.valid_var) / tf.reduce_sum(i.valid_var)
                else:
                    kl = pol_dist.kl_sym(
                        i.valid.policy_old_dist_info_vars,
                        policy_dist_info_valid,
                    )
                    pol_mean_kl = tf.reduce_mean(kl)

            # Calculate vanilla loss
            with tf.name_scope('vanilla_loss'):
                if self.policy.recurrent:
                    ll = pol_dist.log_likelihood_sym(i.action_var,
                                                     policy_dist_info,
                                                     name='log_likelihood')

                    vanilla = ll * adv * i.valid_var
                else:
                    ll = pol_dist.log_likelihood_sym(i.valid.action_var,
                                                     policy_dist_info_valid,
                                                     name='log_likelihood')

                    vanilla = ll * adv_valid

            # Calculate surrogate loss
            with tf.name_scope('surrogate_loss'):
                if self.policy.recurrent:
                    lr = pol_dist.likelihood_ratio_sym(
                        i.action_var,
                        i.policy_old_dist_info_vars,
                        policy_dist_info,
                        name='lr')

                    surrogate = lr * adv * i.valid_var
                else:
                    lr = pol_dist.likelihood_ratio_sym(
                        i.valid.action_var,
                        i.valid.policy_old_dist_info_vars,
                        policy_dist_info_valid,
                        name='lr')

                    surrogate = lr * adv_valid

            # Finalize objective function
            with tf.name_scope('loss'):
                if self._pg_loss == 'vanilla':
                    # VPG uses the vanilla objective
                    obj = tf.identity(vanilla, name='vanilla_obj')
                elif self._pg_loss == 'surrogate':
                    # TRPO uses the standard surrogate objective
                    obj = tf.identity(surrogate, name='surr_obj')
                elif self._pg_loss == 'surrogate_clip':
                    lr_clip = tf.clip_by_value(lr,
                                               1 - self.lr_clip_range,
                                               1 + self.lr_clip_range,
                                               name='lr_clip')
                    if self.policy.recurrent:
                        surr_clip = lr_clip * adv * i.valid_var
                    else:
                        surr_clip = lr_clip * adv_valid
                    obj = tf.minimum(surrogate, surr_clip, name='surr_obj')

                if self._entropy_regularzied:
                    obj += self.policy_ent_coeff * policy_entropy

                # Maximize E[surrogate objective] by minimizing
                # -E_t[surrogate objective]
                if self.policy.recurrent:
                    loss = -tf.reduce_sum(obj) / tf.reduce_sum(i.valid_var)
                else:
                    loss = -tf.reduce_mean(obj)

            # Diagnostic functions
            self.f_policy_kl = compile_function(flatten_inputs(
                self._policy_opt_inputs),
                                                pol_mean_kl,
                                                log_name='f_policy_kl')

            self.f_rewards = compile_function(flatten_inputs(
                self._policy_opt_inputs),
                                              rewards,
                                              log_name='f_rewards')

            returns = discounted_returns(self.discount, self.max_path_length,
                                         rewards)
            self.f_returns = compile_function(flatten_inputs(
                self._policy_opt_inputs),
                                              returns,
                                              log_name='f_returns')

            return loss, pol_mean_kl

    def _build_entropy_term(self, i):
        with tf.name_scope('policy_entropy'):
            if self.policy.recurrent:
                policy_dist_info = self.policy.dist_info_sym(
                    i.obs_var,
                    i.policy_state_info_vars,
                    name='policy_dist_info_2')

                policy_neg_log_likeli = -self.policy.distribution.log_likelihood_sym(  # noqa: E501
                    i.action_var,
                    policy_dist_info,
                    name='policy_log_likeli')

                if self._use_neg_logli_entropy:
                    policy_entropy = policy_neg_log_likeli
                else:
                    policy_entropy = self.policy.distribution.entropy_sym(
                        policy_dist_info)
            else:
                policy_dist_info_flat = self.policy.dist_info_sym(
                    i.flat.obs_var,
                    i.flat.policy_state_info_vars,
                    name='policy_dist_info_flat_2')

                policy_neg_log_likeli_flat = -self.policy.distribution.log_likelihood_sym(  # noqa: E501
                    i.flat.action_var,
                    policy_dist_info_flat,
                    name='policy_log_likeli_flat')

                policy_dist_info_valid = filter_valids_dict(
                    policy_dist_info_flat,
                    i.flat.valid_var,
                    name='policy_dist_info_valid_2')

                policy_neg_log_likeli_valid = -self.policy.distribution.log_likelihood_sym(  # noqa: E501
                    i.valid.action_var,
                    policy_dist_info_valid,
                    name='policy_log_likeli_valid')

                if self._use_neg_logli_entropy:
                    if self._maximum_entropy:
                        policy_entropy = tf.reshape(policy_neg_log_likeli_flat,
                                                    [-1, self.max_path_length])
                    else:
                        policy_entropy = policy_neg_log_likeli_valid
                else:
                    if self._maximum_entropy:
                        policy_entropy_flat = self.policy.distribution.entropy_sym(  # noqa: E501
                            policy_dist_info_flat)
                        policy_entropy = tf.reshape(policy_entropy_flat,
                                                    [-1, self.max_path_length])
                    else:
                        policy_entropy_valid = self.policy.distribution.entropy_sym(  # noqa: E501
                            policy_dist_info_valid)
                        policy_entropy = policy_entropy_valid

            # This prevents entropy from becoming negative for small policy std
            if self._use_softplus_entropy:
                policy_entropy = tf.nn.softplus(policy_entropy)

            if self._stop_entropy_gradient:
                policy_entropy = tf.stop_gradient(policy_entropy)

        self.f_policy_entropy = compile_function(flatten_inputs(
            self._policy_opt_inputs),
                                                 policy_entropy,
                                                 log_name='f_policy_entropy')

        return policy_entropy

    def _fit_baseline(self, samples_data):
        """Update baselines from samples."""
        policy_opt_input_values = self._policy_opt_input_values(samples_data)

        # Augment reward from baselines
        rewards_tensor = self.f_rewards(*policy_opt_input_values)
        returns_tensor = self.f_returns(*policy_opt_input_values)
        returns_tensor = np.squeeze(returns_tensor, -1)

        paths = samples_data['paths']
        valids = samples_data['valids']
        baselines = [path['baselines'] for path in paths]

        # Recompute parts of samples_data
        aug_rewards = []
        aug_returns = []
        for rew, ret, val, path in zip(rewards_tensor, returns_tensor, valids,
                                       paths):
            path['rewards'] = rew[val.astype(np.bool)]
            path['returns'] = ret[val.astype(np.bool)]
            aug_rewards.append(path['rewards'])
            aug_returns.append(path['returns'])
        aug_rewards = concat_tensor_list(aug_rewards)
        aug_returns = concat_tensor_list(aug_returns)
        samples_data['rewards'] = aug_rewards
        samples_data['returns'] = aug_returns

        # Calculate explained variance
        ev = np_tensor_utils.explained_variance_1d(np.concatenate(baselines),
                                                   aug_returns)
        tabular.record('{}/ExplainedVariance'.format(self.baseline.name), ev)

        # Fit baseline
        logger.log('Fitting baseline...')
        if hasattr(self.baseline, 'fit_with_samples'):
            self.baseline.fit_with_samples(paths, samples_data)
        else:
            self.baseline.fit(paths)

    def _policy_opt_input_values(self, samples_data):
        """Map rollout samples to the policy optimizer inputs."""
        policy_state_info_list = [
            samples_data['agent_infos'][k] for k in self.policy.state_info_keys
        ]
        policy_old_dist_info_list = [
            samples_data['agent_infos'][k]
            for k in self.policy.distribution.dist_info_keys
        ]

        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=samples_data['observations'],
            action_var=samples_data['actions'],
            reward_var=samples_data['rewards'],
            baseline_var=samples_data['baselines'],
            valid_var=samples_data['valids'],
            policy_state_info_vars_list=policy_state_info_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_list,
        )

        return flatten_inputs(policy_opt_input_values)

    def _check_entropy_configuration(self, entropy_method, center_adv,
                                     stop_entropy_gradient,
                                     use_neg_logli_entropy, policy_ent_coeff):
        """Check entropy configuration."""
        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
            self._maximum_entropy = True
            self._entropy_regularzied = False
        elif entropy_method == 'regularized':
            self._maximum_entropy = False
            self._entropy_regularzied = True
        elif entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')
            self._maximum_entropy = False
            self._entropy_regularzied = False
        else:
            raise ValueError('Invalid entropy_method')

    def __getstate__(self):
        """Get state."""
        data = self.__dict__.copy()
        del data['_name_scope']
        del data['_policy_opt_inputs']
        del data['f_policy_entropy']
        del data['f_policy_kl']
        del data['f_rewards']
        del data['f_returns']
        return data

    def __setstate__(self, state):
        """Set state."""
        self.__dict__ = state
        self._name_scope = tf.name_scope(self.name)
        self.init_opt()
