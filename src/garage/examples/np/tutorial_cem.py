#!/usr/bin/env python3
"""This is an example to add a Cross Entropy Method algorithm."""
import numpy as np

from garage import EpisodeBatch, log_performance, wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np import discount_cumsum
from garage.sampler import LocalSampler
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer


# pylint: disable=too-few-public-methods
class SimpleCEM:
    """Simple Cross Entropy Method.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (Policy): Action policy.
        sampler (garage.sampler.Sampler): Sampler.

    """

    def __init__(self, env_spec, policy, sampler):
        self.env_spec = env_spec
        self.policy = policy
        self._sampler = sampler
        self.max_episode_length = env_spec.max_episode_length
        self._discount = 0.99
        self._extra_std = 1
        self._extra_decay_time = 100
        self._n_samples = 20
        self._n_best = 1
        self._cur_std = 1
        self._cur_mean = self.policy.get_param_values()
        self._all_avg_returns = []
        self._all_params = [self._cur_mean.copy()]
        self._cur_params = None

    def train(self, trainer):
        """Get samples and train the policy.

        Args:
            trainer (Trainer): Trainer.

        """
        for epoch in trainer.step_epochs():
            samples = trainer.obtain_samples(epoch)
            log_performance(epoch,
                            EpisodeBatch.from_list(self.env_spec, samples),
                            self._discount)
            self._train_once(epoch, samples)

    def _train_once(self, epoch, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            epoch (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            float: The average return of epoch cycle.

        """
        returns = []
        for path in paths:
            returns.append(discount_cumsum(path['rewards'], self._discount))
        avg_return = np.mean(np.concatenate(returns))
        self._all_avg_returns.append(avg_return)

        if (epoch + 1) % self._n_samples == 0:
            avg_rtns = np.array(self._all_avg_returns)
            best_inds = np.argsort(-avg_rtns)[:self._n_best]
            best_params = np.array(self._all_params)[best_inds]
            self._cur_mean = best_params.mean(axis=0)
            self._cur_std = best_params.std(axis=0)
            self.policy.set_param_values(self._cur_mean)
            avg_return = max(self._all_avg_returns)
            self._all_avg_returns.clear()
            self._all_params.clear()

        self._cur_params = self._sample_params(epoch)
        self._all_params.append(self._cur_params.copy())
        self.policy.set_param_values(self._cur_params)

        return avg_return

    def _sample_params(self, epoch):
        """Return sample parameters.

        Args:
            epoch (int): Epoch number.

        Returns:
            np.ndarray: A numpy array of parameter values.

        """
        extra_var_mult = max(1.0 - epoch / self._extra_decay_time, 0)
        sample_std = np.sqrt(
            np.square(self._cur_std) +
            np.square(self._extra_std) * extra_var_mult)

        return np.random.standard_normal(len(
            self._cur_mean)) * sample_std + self._cur_mean


@wrap_experiment
def tutorial_cem(ctxt=None):
    """Train CEM with Cartpole-v1 environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.

    """
    set_seed(100)
    with TFTrainer(ctxt) as trainer:
        env = GymEnv('CartPole-v1')
        policy = CategoricalMLPPolicy(env.spec)
        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True)
        algo = SimpleCEM(env.spec, policy, sampler)
        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=1000)


tutorial_cem()
