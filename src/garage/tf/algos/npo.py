"""Natural Policy Gradient Optimization."""
# pylint: disable=wrong-import-order
# yapf: disable
import collections

from dowel import logger, tabular
import numpy as np
import tensorflow as tf

from garage import log_performance, make_optimizer
from garage.np import explained_variance_1d, pad_batch_array
from garage.np.algos import RLAlgorithm
from garage.tf import (center_advs, compile_function, compute_advantages,
                       discounted_returns, flatten_inputs, graph_inputs,
                       positive_advs)
from garage.tf.optimizers import FirstOrderOptimizer

# yapf: enable


class NPO(RLAlgorithm):
    """Natural Policy Gradient Optimization.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        sampler (garage.sampler.Sampler): Sampler.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
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
                 sampler,
                 scope=None,
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
                 name='NPO'):
        self.policy = policy
        self._scope = scope
        self.max_episode_length = env_spec.max_episode_length
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
        self._use_softplus_entropy = use_softplus_entropy
        self._use_neg_logli_entropy = use_neg_logli_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._pg_loss = pg_loss
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = FirstOrderOptimizer

        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          use_neg_logli_entropy,
                                          policy_ent_coeff)

        if pg_loss not in ['vanilla', 'surrogate', 'surrogate_clip']:
            raise ValueError('Invalid pg_loss')

        self._optimizer = make_optimizer(optimizer, **optimizer_args)
        self._lr_clip_range = float(lr_clip_range)
        self._max_kl_step = float(max_kl_step)
        self._policy_ent_coeff = float(policy_ent_coeff)

        self._f_rewards = None
        self._f_returns = None
        self._f_policy_kl = None
        self._f_policy_entropy = None
        self._policy_network = None
        self._old_policy_network = None

        self._episode_reward_mean = collections.deque(maxlen=100)

        self._sampler = sampler

        self._init_opt()

    def _init_opt(self):
        """Initialize optimizater."""
        pol_loss_inputs, pol_opt_inputs = self._build_inputs()
        self._policy_opt_inputs = pol_opt_inputs

        pol_loss, pol_kl = self._build_policy_loss(pol_loss_inputs)
        self._optimizer.update_opt(loss=pol_loss,
                                   target=self.policy,
                                   leq_constraint=(pol_kl, self._max_kl_step),
                                   inputs=flatten_inputs(
                                       self._policy_opt_inputs),
                                   constraint_name='mean_kl')

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer, which rovides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in trainer.step_epochs():
            trainer.step_episode = trainer.obtain_episodes(trainer.step_itr)
            last_return = self._train_once(trainer.step_itr,
                                           trainer.step_episode)
            trainer.step_itr += 1

        return last_return

    def _train_once(self, itr, episodes):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.

        Returns:
            numpy.float64: Average return.

        """
        # -- Stage: Calculate and pad baselines
        obs = [
            self._baseline.predict({'observations': obs})
            for obs in episodes.observations_list
        ]
        baselines = pad_batch_array(np.concatenate(obs), episodes.lengths,
                                    self.max_episode_length)

        # -- Stage: Run and calculate performance of the algorithm
        undiscounted_returns = log_performance(itr,
                                               episodes,
                                               discount=self._discount)
        self._episode_reward_mean.extend(undiscounted_returns)
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self._episode_reward_mean))

        logger.log('Optimizing policy...')
        self._optimize_policy(episodes, baselines)

        return np.mean(undiscounted_returns)

    def _optimize_policy(self, episodes, baselines):
        """Optimize policy.

        Args:
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.

        """
        policy_opt_input_values = self._policy_opt_input_values(
            episodes, baselines)
        logger.log('Computing loss before')
        loss_before = self._optimizer.loss(policy_opt_input_values)
        logger.log('Computing KL before')
        policy_kl_before = self._f_policy_kl(*policy_opt_input_values)
        logger.log('Optimizing')
        self._optimizer.optimize(policy_opt_input_values)
        logger.log('Computing KL after')
        policy_kl = self._f_policy_kl(*policy_opt_input_values)
        logger.log('Computing loss after')
        loss_after = self._optimizer.loss(policy_opt_input_values)
        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)
        pol_ent = self._f_policy_entropy(*policy_opt_input_values)
        ent = np.sum(pol_ent) / np.sum(episodes.lengths)
        tabular.record('{}/Entropy'.format(self.policy.name), ent)
        tabular.record('{}/Perplexity'.format(self.policy.name), np.exp(ent))
        returns = self._fit_baseline_with_data(episodes, baselines)

        ev = explained_variance_1d(baselines, returns, episodes.valids)

        tabular.record('{}/ExplainedVariance'.format(self._baseline.name), ev)
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
            obs_var = observation_space.to_tf_placeholder(name='obs',
                                                          batch_dims=2)
            action_var = action_space.to_tf_placeholder(name='action',
                                                        batch_dims=2)
            reward_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None, None],
                                                  name='reward')
            valid_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=[None, None],
                                                 name='valid')
            baseline_var = tf.compat.v1.placeholder(tf.float32,
                                                    shape=[None, None],
                                                    name='baseline')

            policy_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name=k)
                for k, shape in self.policy.state_info_specs
            }
            policy_state_info_vars_list = [
                policy_state_info_vars[k] for k in self.policy.state_info_keys
            ]

        augmented_obs_var = obs_var
        for k in self.policy.state_info_keys:
            extra_state_var = policy_state_info_vars[k]
            extra_state_var = tf.cast(extra_state_var, tf.float32)
            augmented_obs_var = tf.concat([augmented_obs_var, extra_state_var],
                                          -1)

        self._policy_network = self.policy.build(augmented_obs_var,
                                                 name='policy')
        self._old_policy_network = self._old_policy.build(augmented_obs_var,
                                                          name='policy')

        policy_loss_inputs = graph_inputs(
            'PolicyLossInputs',
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            valid_var=valid_var,
            policy_state_info_vars=policy_state_info_vars,
        )
        policy_opt_inputs = graph_inputs(
            'PolicyOptInputs',
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            valid_var=valid_var,
            policy_state_info_vars_list=policy_state_info_vars_list,
        )

        return policy_loss_inputs, policy_opt_inputs

    # pylint: disable=too-many-branches, too-many-statements
    def _build_policy_loss(self, i):
        """Build policy loss and other output tensors.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy loss.
            tf.Tensor: Mean policy KL divergence.

        """
        policy_entropy = self._build_entropy_term(i)
        rewards = i.reward_var

        if self._maximum_entropy:
            with tf.name_scope('augmented_rewards'):
                rewards = i.reward_var + (self._policy_ent_coeff *
                                          policy_entropy)

        with tf.name_scope('policy_loss'):
            adv = compute_advantages(self._discount,
                                     self._gae_lambda,
                                     self.max_episode_length,
                                     i.baseline_var,
                                     rewards,
                                     name='adv')

            adv = tf.reshape(adv, [-1, self.max_episode_length])
            # Optionally normalize advantages
            eps = tf.constant(1e-8, dtype=tf.float32)
            if self._center_adv:
                adv = center_advs(adv, axes=[0], eps=eps)

            if self._positive_adv:
                adv = positive_advs(adv, eps)

            old_policy_dist = self._old_policy_network.dist
            policy_dist = self._policy_network.dist

            with tf.name_scope('kl'):
                kl = old_policy_dist.kl_divergence(policy_dist)
                pol_mean_kl = tf.reduce_mean(kl)

            # Calculate vanilla loss
            with tf.name_scope('vanilla_loss'):
                ll = policy_dist.log_prob(i.action_var, name='log_likelihood')
                vanilla = ll * adv

            # Calculate surrogate loss
            with tf.name_scope('surrogate_loss'):
                lr = tf.exp(ll - old_policy_dist.log_prob(i.action_var))
                surrogate = lr * adv

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
                                               1 - self._lr_clip_range,
                                               1 + self._lr_clip_range,
                                               name='lr_clip')
                    surr_clip = lr_clip * adv
                    obj = tf.minimum(surrogate, surr_clip, name='surr_obj')

                if self._entropy_regularzied:
                    obj += self._policy_ent_coeff * policy_entropy

                # filter only the valid values
                obj = tf.boolean_mask(obj, i.valid_var)
                # Maximize E[surrogate objective] by minimizing
                # -E_t[surrogate objective]
                loss = -tf.reduce_mean(obj)

            # Diagnostic functions
            self._f_policy_kl = tf.compat.v1.get_default_session(
            ).make_callable(pol_mean_kl,
                            feed_list=flatten_inputs(self._policy_opt_inputs))

            self._f_rewards = tf.compat.v1.get_default_session().make_callable(
                rewards, feed_list=flatten_inputs(self._policy_opt_inputs))

            returns = discounted_returns(self._discount,
                                         self.max_episode_length, rewards)
            self._f_returns = tf.compat.v1.get_default_session().make_callable(
                returns, feed_list=flatten_inputs(self._policy_opt_inputs))

            return loss, pol_mean_kl

    def _build_entropy_term(self, i):
        """Build policy entropy tensor.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy entropy.

        """
        pol_dist = self._policy_network.dist

        with tf.name_scope('policy_entropy'):
            if self._use_neg_logli_entropy:
                policy_entropy = -pol_dist.log_prob(i.action_var,
                                                    name='policy_log_likeli')
            else:
                policy_entropy = pol_dist.entropy()

            # This prevents entropy from becoming negative for small policy std
            if self._use_softplus_entropy:
                policy_entropy = tf.nn.softplus(policy_entropy)

            if self._stop_entropy_gradient:
                policy_entropy = tf.stop_gradient(policy_entropy)

        # dense form, match the shape of advantage
        policy_entropy = tf.reshape(policy_entropy,
                                    [-1, self.max_episode_length])

        self._f_policy_entropy = compile_function(
            flatten_inputs(self._policy_opt_inputs), policy_entropy)

        return policy_entropy

    def _fit_baseline_with_data(self, episodes, baselines):
        """Update baselines from samples.

        Args:
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.

        Returns:
            np.ndarray: Augment returns.

        """
        policy_opt_input_values = self._policy_opt_input_values(
            episodes, baselines)

        returns_tensor = self._f_returns(*policy_opt_input_values)
        returns_tensor = np.squeeze(returns_tensor, -1)

        paths = []
        valids = episodes.valids
        observations = episodes.padded_observations

        # Compute returns
        for ret, val, ob in zip(returns_tensor, valids, observations):
            returns = ret[val.astype(np.bool)]
            obs = ob[val.astype(np.bool)]
            paths.append(dict(observations=obs, returns=returns))

        # Fit baseline
        logger.log('Fitting baseline...')
        self._baseline.fit(paths)
        return returns_tensor

    def _policy_opt_input_values(self, episodes, baselines):
        """Map episode samples to the policy optimizer inputs.

        Args:
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.

        Returns:
            list(np.ndarray): Flatten policy optimization input values.

        """
        agent_infos = episodes.padded_agent_infos
        policy_state_info_list = [
            agent_infos[k] for k in self.policy.state_info_keys
        ]

        actions = [
            self._env_spec.action_space.flatten_n(act)
            for act in episodes.actions_list
        ]
        padded_actions = pad_batch_array(np.concatenate(actions),
                                         episodes.lengths,
                                         self.max_episode_length)

        # pylint: disable=unexpected-keyword-arg
        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=episodes.padded_observations,
            action_var=padded_actions,
            reward_var=episodes.padded_rewards,
            baseline_var=baselines,
            valid_var=episodes.valids,
            policy_state_info_vars_list=policy_state_info_list,
        )

        return flatten_inputs(policy_opt_input_values)

    def _check_entropy_configuration(self, entropy_method, center_adv,
                                     stop_entropy_gradient,
                                     use_neg_logli_entropy, policy_ent_coeff):
        """Check entropy configuration.

        Args:
            entropy_method (str): A string from: 'max', 'regularized',
                'no_entropy'. The type of entropy method to use. 'max' adds the
                dense entropy to the reward for each time step. 'regularized'
                adds the mean entropy to the surrogate objective. See
                https://arxiv.org/abs/1805.00909 for more details.
            center_adv (bool): Whether to rescale the advantages
                so that they have mean 0 and standard deviation 1.
            stop_entropy_gradient (bool): Whether to stop the entropy gradient.
            use_neg_logli_entropy (bool): Whether to estimate the entropy as
                the negative log likelihood of the action.
            policy_ent_coeff (float): The coefficient of the policy entropy.
                Setting it to zero would mean no entropy regularization.

        Raises:
            ValueError: If center_adv is True when entropy_method is max.
            ValueError: If stop_gradient is False when entropy_method is max.
            ValueError: If policy_ent_coeff is non-zero when there is
                no entropy method.
            ValueError: If entropy_method is not one of 'max', 'regularized',
                'no_entropy'.

        """
        del use_neg_logli_entropy

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
        """Parameters to save in snapshot.

        Returns:
            dict: Parameters to save.

        """
        data = self.__dict__.copy()
        del data['_name_scope']
        del data['_policy_opt_inputs']
        del data['_f_policy_entropy']
        del data['_f_policy_kl']
        del data['_f_rewards']
        del data['_f_returns']
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
        self._init_opt()
