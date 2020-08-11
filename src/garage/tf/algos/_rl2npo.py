"""Natural Policy Gradient Optimization."""
from dowel import logger, tabular
import numpy as np

from garage.misc import tensor_utils as np_tensor_utils
from garage.tf.algos import NPO


class RL2NPO(NPO):
    """Natural Policy Gradient Optimization.

    This is specific for RL^2
    (https://arxiv.org/pdf/1611.02779.pdf).

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
        fit_baseline (str): Either 'before' or 'after'. See above docstring for
            a more detail explanation. Currently it only supports 'before'.
        name (str): The name of the algorithm.

    """

    def optimize_policy(self, samples_data):
        """Optimize policy.

        Args:
            samples_data (dict): Processed sample data.
                See garage.tf.paths_to_tensors() for details.

        """
        self._fit_baseline_with_data(samples_data)
        samples_data['baselines'] = self._get_baseline_prediction(samples_data)

        policy_opt_input_values = self._policy_opt_input_values(samples_data)
        # Train policy network
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
        tabular.record('{}/Entropy'.format(self.policy.name), np.mean(pol_ent))

        ev = np_tensor_utils.explained_variance_1d(samples_data['baselines'],
                                                   samples_data['returns'],
                                                   samples_data['valids'])
        tabular.record('{}/ExplainedVariance'.format(self._baseline.name), ev)
        self._old_policy.parameters = self.policy.parameters

    def _get_baseline_prediction(self, samples_data):
        """Get baseline prediction.

        Args:
            samples_data (dict): Processed sample data.
                See garage.tf.paths_to_tensors() for details.

        Returns:
            np.ndarray: Baseline prediction, with shape
                :math:`(N, max_episode_length * episode_per_task)`.

        """
        paths = samples_data['paths']
        baselines = [self._baseline.predict(path) for path in paths]
        return np_tensor_utils.pad_tensor_n(baselines, self.max_episode_length)
