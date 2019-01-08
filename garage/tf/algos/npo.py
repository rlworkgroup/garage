"""Natural Policy Gradient Optimization."""
from enum import Enum
from enum import unique

import numpy as np
import tensorflow as tf

from garage.misc import logger
from garage.misc import special
from garage.misc.overrides import overrides
from garage.tf.algos import BatchPolopt
from garage.tf.misc import tensor_utils
from garage.tf.misc.tensor_utils import compute_advantages
from garage.tf.misc.tensor_utils import discounted_returns
from garage.tf.misc.tensor_utils import filter_valids
from garage.tf.misc.tensor_utils import filter_valids_dict
from garage.tf.misc.tensor_utils import flatten_batch
from garage.tf.misc.tensor_utils import flatten_batch_dict
from garage.tf.misc.tensor_utils import flatten_inputs
from garage.tf.misc.tensor_utils import graph_inputs
from garage.tf.optimizers import LbfgsOptimizer


@unique
class PGLoss(Enum):
    VANILLA = "vanilla"
    SURROGATE = "surrogate"
    SURROGATE_CLIP = "surrogate_clip"


class NPO(BatchPolopt):
    """
    Natural Policy Gradient Optimization.

    Attributes:
        name(str): The name of the algorithm.
        lr_clip_range(float): The limit on the likelihood ratio between
            policies, as in PPO.
        max_kl_step(float): The maximum KL divergence between old and new
            policies, as in TRPO.
        policy_ent_coeff(float): The coefficient of the policy entropy.
        optimizer(float): The optimizer of the algorithm.
    """

    def __init__(self,
                 pg_loss=PGLoss.SURROGATE,
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 name="NPO",
                 policy=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=True,
                 use_neg_logli_entropy=True,
                 stop_entropy_gradient=True,
                 **kwargs):
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

        with self._name_scope:
            self.optimizer = optimizer(**optimizer_args)
            self.lr_clip_range = float(lr_clip_range)
            self.max_kl_step = float(max_kl_step)
            self.policy_ent_coeff = float(policy_ent_coeff)

        super().__init__(policy=policy, **kwargs)

    @overrides
    def init_opt(self):
        pol_loss_inputs, pol_opt_inputs = self._build_inputs()
        self._policy_opt_inputs = pol_opt_inputs

        pol_loss, pol_kl = self._build_policy_loss(pol_loss_inputs)
        self.optimizer.update_opt(
            loss=pol_loss,
            target=self.policy,
            leq_constraint=(pol_kl, self.max_kl_step),
            inputs=flatten_inputs(self._policy_opt_inputs),
            constraint_name="mean_kl")

        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        policy_opt_input_values = self._policy_opt_input_values(samples_data)

        # Train policy network
        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(policy_opt_input_values)
        logger.log("Computing KL before")
        policy_kl_before = self.f_policy_kl(*policy_opt_input_values)
        logger.log("Optimizing")
        self.optimizer.optimize(policy_opt_input_values)
        logger.log("Computing KL after")
        policy_kl = self.f_policy_kl(*policy_opt_input_values)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(policy_opt_input_values)
        logger.record_tabular("{}/LossBefore".format(self.policy.name),
                              loss_before)
        logger.record_tabular("{}/LossAfter".format(self.policy.name),
                              loss_after)
        logger.record_tabular("{}/dLoss".format(self.policy.name),
                              loss_before - loss_after)
        logger.record_tabular("{}/KLBefore".format(self.policy.name),
                              policy_kl_before)
        logger.record_tabular("{}/KL".format(self.policy.name), policy_kl)

        pol_ent = self.f_policy_entropy(*policy_opt_input_values)
        logger.record_tabular("{}/Entropy".format(self.policy.name),
                              np.mean(pol_ent))

        num_traj = self.batch_size // self.max_path_length
        actions = samples_data["actions"][:num_traj, ...]
        logger.record_histogram("{}/Actions".format(self.policy.name), actions)

        self._fit_baseline(samples_data)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

    def _build_inputs(self):
        observation_space = self.policy.observation_space
        action_space = self.policy.action_space

        policy_dist = self.policy.distribution

        with tf.name_scope("inputs"):
            obs_var = observation_space.new_tensor_variable(
                name="obs", extra_dims=2)
            action_var = action_space.new_tensor_variable(
                name="action", extra_dims=2)
            reward_var = tensor_utils.new_tensor(
                name="reward", ndim=2, dtype=tf.float32)
            valid_var = tf.placeholder(
                tf.float32, shape=[None, None], name="valid")
            baseline_var = tensor_utils.new_tensor(
                name="baseline", ndim=2, dtype=tf.float32)

            policy_state_info_vars = {
                k: tf.placeholder(
                    tf.float32, shape=[None] * 2 + list(shape), name=k)
                for k, shape in self.policy.state_info_specs
            }
            policy_state_info_vars_list = [
                policy_state_info_vars[k] for k in self.policy.state_info_keys
            ]

            # old policy distribution
            policy_old_dist_info_vars = {
                k: tf.placeholder(
                    tf.float32,
                    shape=[None] * 2 + list(shape),
                    name="policy_old_%s" % k)
                for k, shape in policy_dist.dist_info_specs
            }
            policy_old_dist_info_vars_list = [
                policy_old_dist_info_vars[k]
                for k in policy_dist.dist_info_keys
            ]

            # flattened view
            with tf.name_scope("flat"):
                obs_flat = flatten_batch(obs_var, name="obs_flat")
                action_flat = flatten_batch(action_var, name="action_flat")
                reward_flat = flatten_batch(reward_var, name="reward_flat")
                valid_flat = flatten_batch(valid_var, name="valid_flat")
                policy_state_info_vars_flat = flatten_batch_dict(
                    policy_state_info_vars, name="policy_state_info_vars_flat")
                policy_old_dist_info_vars_flat = flatten_batch_dict(
                    policy_old_dist_info_vars,
                    name="policy_old_dist_info_vars_flat")

            # valid view
            with tf.name_scope("valid"):
                action_valid = filter_valids(
                    action_flat, valid_flat, name="action_valid")
                policy_state_info_vars_valid = filter_valids_dict(
                    policy_state_info_vars_flat,
                    valid_flat,
                    name="policy_state_info_vars_valid")
                policy_old_dist_info_vars_valid = filter_valids_dict(
                    policy_old_dist_info_vars_flat,
                    valid_flat,
                    name="policy_old_dist_info_vars_valid")

        # policy loss and optimizer inputs
        pol_flat = graph_inputs(
            "PolicyLossInputsFlat",
            obs_var=obs_flat,
            action_var=action_flat,
            reward_var=reward_flat,
            valid_var=valid_flat,
            policy_state_info_vars=policy_state_info_vars_flat,
            policy_old_dist_info_vars=policy_old_dist_info_vars_flat,
        )
        pol_valid = graph_inputs(
            "PolicyLossInputsValid",
            action_var=action_valid,
            policy_state_info_vars=policy_state_info_vars_valid,
            policy_old_dist_info_vars=policy_old_dist_info_vars_valid,
        )
        policy_loss_inputs = graph_inputs(
            "PolicyLossInputs",
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
            "PolicyOptInputs",
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

        with tf.name_scope("augmented_rewards"):
            rewards = i.reward_var + (self.policy_ent_coeff * policy_entropy)

        with tf.name_scope("policy_loss"):
            advantages = compute_advantages(
                self.discount,
                self.gae_lambda,
                self.max_path_length,
                i.baseline_var,
                rewards,
                name="advantages")

            adv_flat = flatten_batch(advantages, name="adv_flat")
            adv_valid = filter_valids(
                adv_flat, i.flat.valid_var, name="adv_valid")

            if self.policy.recurrent:
                advantages = tf.reshape(advantages, [-1, self.max_path_length])

            # Optionally normalize advantages
            eps = tf.constant(1e-8, dtype=tf.float32)
            if self.center_adv:
                with tf.name_scope("center_adv"):
                    mean, var = tf.nn.moments(adv_valid, axes=[0])
                    adv_valid = tf.nn.batch_normalization(
                        adv_valid, mean, var, 0, 1, eps)
            if self.positive_adv:
                with tf.name_scope("positive_adv"):
                    m = tf.reduce_min(adv_valid)
                    adv_valid = (adv_valid - m) + eps

            if self.policy.recurrent:
                policy_dist_info = self.policy.dist_info_sym(
                    i.obs_var,
                    i.policy_state_info_vars,
                    name="policy_dist_info")
            else:
                policy_dist_info_flat = self.policy.dist_info_sym(
                    i.flat.obs_var,
                    i.flat.policy_state_info_vars,
                    name="policy_dist_info_flat")

                policy_dist_info_valid = filter_valids_dict(
                    policy_dist_info_flat,
                    i.flat.valid_var,
                    name="policy_dist_info_valid")

            # Calculate loss function and KL divergence
            with tf.name_scope("kl"):
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
            with tf.name_scope("vanilla_loss"):
                if self.policy.recurrent:
                    ll = pol_dist.log_likelihood_sym(
                        i.action_var, policy_dist_info, name="log_likelihood")

                    vanilla = ll * advantages * i.valid_var
                else:
                    ll = pol_dist.log_likelihood_sym(
                        i.valid.action_var,
                        policy_dist_info_valid,
                        name="log_likelihood")

                    vanilla = ll * adv_valid

            # Calculate surrogate loss
            with tf.name_scope("surrogate_loss"):
                if self.policy.recurrent:
                    lr = pol_dist.likelihood_ratio_sym(
                        i.action_var,
                        i.policy_old_dist_info_vars,
                        policy_dist_info,
                        name="lr")

                    surrogate = lr * advantages * i.valid_var
                else:
                    lr = pol_dist.likelihood_ratio_sym(
                        i.valid.action_var,
                        i.valid.policy_old_dist_info_vars,
                        policy_dist_info_valid,
                        name="lr")

                    surrogate = lr * adv_valid

            # Finalize objective function
            with tf.name_scope("loss"):
                if self._pg_loss == PGLoss.VANILLA:
                    # VPG uses the vanilla objective
                    obj = tf.identity(vanilla, name="vanilla_obj")
                elif self._pg_loss == PGLoss.SURROGATE:
                    # TRPO uses the standard surrogate objective
                    obj = tf.identity(surrogate, name="surr_obj")
                elif self._pg_loss == PGLoss.SURROGATE_CLIP:
                    lr_clip = tf.clip_by_value(
                        lr,
                        1 - self.lr_clip_range,
                        1 + self.lr_clip_range,
                        name="lr_clip")
                    if self.policy.recurrent:
                        surr_clip = lr_clip * advantages * i.valid_var
                    else:
                        surr_clip = lr_clip * adv_valid
                    obj = tf.minimum(surrogate, surr_clip, name="surr_obj")
                else:
                    raise NotImplementedError("Unknown PGLoss")

                # Maximize E[surrogate objective] by minimizing
                # -E_t[surrogate objective]
                if self.policy.recurrent:
                    loss = -tf.reduce_sum(obj) / tf.reduce_sum(i.valid_var)
                else:
                    loss = -tf.reduce_mean(obj)

            # Diagnostic functions
            self.f_policy_kl = tensor_utils.compile_function(
                flatten_inputs(self._policy_opt_inputs),
                pol_mean_kl,
                log_name="f_policy_kl")

            self.f_rewards = tensor_utils.compile_function(
                flatten_inputs(self._policy_opt_inputs),
                rewards,
                log_name="f_rewards")

            returns = discounted_returns(self.discount, self.max_path_length,
                                         rewards)
            self.f_returns = tensor_utils.compile_function(
                flatten_inputs(self._policy_opt_inputs),
                returns,
                log_name="f_returns")

            return loss, pol_mean_kl

    def _build_entropy_term(self, i):
        with tf.name_scope("policy_entropy"):
            if self.policy.recurrent:
                policy_dist_info_flat = self.policy.dist_info_sym(
                    i.obs_var,
                    i.policy_state_info_vars,
                    name="policy_dist_info")
                policy_neg_log_likeli_flat = self.policy.distribution.log_likelihood_sym(  # noqa: E501
                    i.action_var,
                    policy_dist_info_flat,
                    name="policy_log_likeli")
            else:
                policy_dist_info_flat = self.policy.dist_info_sym(
                    i.flat.obs_var,
                    i.flat.policy_state_info_vars,
                    name="policy_dist_info_flat")
                policy_neg_log_likeli_flat = self.policy.distribution.log_likelihood_sym(  # noqa: E501
                    i.flat.action_var,
                    policy_dist_info_flat,
                    name="policy_log_likeli")

            if self._use_neg_logli_entropy:
                policy_entropy_flat = policy_neg_log_likeli_flat
            else:
                policy_entropy_flat = self.policy.distribution.entropy_sym(
                    policy_dist_info_flat)

            policy_entropy = tf.reshape(policy_entropy_flat,
                                        [-1, self.max_path_length])

            # This prevents entropy from becoming negative for small policy std
            if self._use_softplus_entropy:
                policy_entropy = tf.nn.softplus(policy_entropy)

            policy_entropy = policy_entropy * i.valid_var

            if self._stop_entropy_gradient:
                policy_entropy = tf.stop_gradient(policy_entropy)

        self.f_policy_entropy = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs),
            policy_entropy,
            log_name="f_policy_entropy")

        return policy_entropy

    def _fit_baseline(self, samples_data):
        """ Update baselines from samples. """

        policy_opt_input_values = self._policy_opt_input_values(samples_data)

        # Augment reward from baselines
        rewards_tensor = self.f_rewards(*policy_opt_input_values)
        returns_tensor = self.f_returns(*policy_opt_input_values)
        returns_tensor = np.squeeze(returns_tensor)

        paths = samples_data["paths"]
        valids = samples_data["valids"]
        baselines = [path["baselines"] for path in paths]

        # Recompute parts of samples_data
        aug_rewards = []
        aug_returns = []
        for rew, ret, val, path in zip(rewards_tensor, returns_tensor, valids,
                                       paths):
            path["rewards"] = rew[val.astype(np.bool)]
            path["returns"] = ret[val.astype(np.bool)]
            aug_rewards.append(path["rewards"])
            aug_returns.append(path["returns"])
        aug_rewards = tensor_utils.concat_tensor_list(aug_rewards)
        aug_returns = tensor_utils.concat_tensor_list(aug_returns)
        samples_data["rewards"] = aug_rewards
        samples_data["returns"] = aug_returns

        # Calculate explained variance
        ev = special.explained_variance_1d(
            np.concatenate(baselines), aug_returns)
        logger.record_tabular("Baseline/ExplainedVariance", ev)

        # Fit baseline
        logger.log("Fitting baseline...")
        if hasattr(self.baseline, "fit_with_samples"):
            self.baseline.fit_with_samples(paths, samples_data)
        else:
            self.baseline.fit(paths)

    def _policy_opt_input_values(self, samples_data):
        """ Map rollout samples to the policy optimizer inputs """
        policy_state_info_list = [
            samples_data["agent_infos"][k] for k in self.policy.state_info_keys
        ]
        policy_old_dist_info_list = [
            samples_data["agent_infos"][k]
            for k in self.policy.distribution.dist_info_keys
        ]

        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=samples_data["observations"],
            action_var=samples_data["actions"],
            reward_var=samples_data["rewards"],
            baseline_var=samples_data["baselines"],
            valid_var=samples_data["valids"],
            policy_state_info_vars_list=policy_state_info_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_list,
        )

        return flatten_inputs(policy_opt_input_values)
