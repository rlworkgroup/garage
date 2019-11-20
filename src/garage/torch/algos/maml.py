import copy
import collections

from dowel import tabular
import numpy as np
import torch

from garage.misc import tensor_utils
from garage.sampler import OnPolicyVectorizedSampler
from garage.tf.samplers import BatchSampler
from garage.torch.algos import VPG
from garage.torch.optimizers import ConjugateGradientOptimizer
from garage.torch.optimizers import DiffSGD
from garage.torch.utils import update_module_params, zero_grad


class MAML:

    def __init__(
        self,
        env,
        policy,
        baseline,
        meta_batch_size=40,
        lr=0.1,
        max_kl_step=0.01,
        num_grad_updates=1,
        inner_algo=None,
        meta_optimizer=None
    ):
        if inner_algo is None:
            inner_algo = VPG(env.spec,
                              policy,
                              baseline)
        if meta_optimizer is None:
            meta_optimizer = ConjugateGradientOptimizer(
                                policy.parameters(),
                                max_constraint_value=max_kl_step)
        if policy.vectorized:
            self.sampler_cls = OnPolicyVectorizedSampler
        else:
            self.sampler_cls = BatchSampler

        self.policy = policy
        self.baseline = baseline
        self.max_path_length = inner_algo.max_path_length
        self._env = env
        self._num_grad_updates = num_grad_updates
        self._meta_batch_size = meta_batch_size
        self._inner_algo = inner_algo
        self._inner_optimizer = DiffSGD(self.policy, lr=lr)
        self._meta_optimizer = meta_optimizer
        self._episode_reward_mean = [
            collections.deque(maxlen=100) for _ in range(num_grad_updates+1)
        ]

    def train(self, runner):
        last_return = None

        for _ in runner.step_epochs():
            tasks = self._env.sample_tasks(self._meta_batch_size)
            print("Obtaining samples....")
            all_samples, all_adapt_params = self._obtain_samples(runner, tasks)
            print("Training....")
            last_return = self.train_once(runner, all_samples, all_adapt_params)
            runner.step_itr += 1

        return last_return

    def train_once(self, runner, all_samples, all_adapt_params):
        # TODO: Create a named tuple to store sample infos
        
        kl_before = self._compute_kl_constraint(runner.step_itr,
                                                all_samples,
                                                all_adapt_params, set_grad=False)
        
        meta_objective = self._compute_meta_loss(runner.step_itr,
                                                 all_samples,
                                                 all_adapt_params)

        self._meta_optimizer.zero_grad()
        meta_objective.backward()

        self._meta_optimize(runner.step_itr, all_samples, all_adapt_params)

        # Log
        loss_after = self._compute_meta_loss(runner.step_itr,
                                                all_samples,
                                                all_adapt_params, set_grad=False)
        kl_after = self._compute_kl_constraint(runner.step_itr,
                                                all_samples,
                                                all_adapt_params, set_grad=False)

        with torch.no_grad():
            policy_entropy = self._compute_policy_entropy([task_samples[0] for task_samples in all_samples])
            average_return = self._log(runner.step_itr, all_samples,
                                        meta_objective.item(),
                                        loss_after.item(),
                                        kl_before.item(), kl_after.item(),
                                        policy_entropy.mean().item())

        return average_return

    def _obtain_samples(self, runner, tasks):
        all_samples = [[] for _ in range(len(tasks))]
        all_adapt_params = []
        theta = dict(self.policy.named_parameters())

        for i, task in enumerate(tasks):
            self._env.set_task(task)

            for j in range(self._num_grad_updates + 1):
                paths = runner.obtain_samples(runner.step_itr)
                valids, obs, actions, rewards, baselines = self._process_samples(runner.step_itr, paths)
                all_samples[i].append((paths, valids, obs, actions, rewards, baselines))

                # Skip last iteration
                if j != self._num_grad_updates:
                    self._optimize(runner.step_itr, paths, valids, obs, actions, rewards, baselines, set_grad=False)

            all_adapt_params.append(dict(self.policy.named_parameters()))
            update_module_params(self.policy, theta)

        return all_samples, all_adapt_params

    def _optimize(self, itr, paths, valids, obs, actions, rewards, baselines, set_grad=True):
        loss = self._compute_loss(itr, paths, valids, obs, actions, rewards, baselines)

        # Update policy parameters with one SGD step
        zero_grad(self.policy.parameters())
        loss.backward(create_graph=set_grad)
        
        with torch.set_grad_enabled(set_grad):
            self._inner_optimizer.step()

    def _meta_optimize(self, itr, all_samples, all_adapt_params):
        self._meta_optimizer.step(
            f_loss=lambda: self._compute_meta_loss(itr, all_samples,
                                                   all_adapt_params, set_grad=False),
            f_constraint=lambda: self._compute_kl_constraint(itr, all_samples,
                                                             all_adapt_params))

    def _compute_loss(self, itr, paths, valids, obs, actions, rewards, baselines):        
        loss = self._inner_algo.compute_loss(itr, paths, valids, obs, actions, rewards, baselines)
        return loss

    def _compute_meta_loss(self, itr, all_samples, all_adapt_params, set_grad=True):
        theta = dict(self.policy.named_parameters())
        old_theta = dict(self._old_policy.named_parameters())
        
        losses = []
        for samples, adapt_params in zip(all_samples, all_adapt_params):
            for i in range(self._num_grad_updates):
                self._optimize(itr, *samples[i], set_grad=set_grad)

            update_module_params(self._old_policy, adapt_params)
            with torch.set_grad_enabled(set_grad):
                loss = self._compute_loss(itr, *samples[-1])
            losses.append(loss)

            zero_grad(theta.values())
            update_module_params(self.policy, theta)
            update_module_params(self._old_policy, old_theta)

        return torch.stack(losses).mean()

    def _compute_kl_constraint(self, itr, all_samples, all_adapt_params, set_grad=True):
        theta = dict(self.policy.named_parameters())
        old_theta = dict(self._old_policy.named_parameters())

        kls = []
        for samples, adapt_params in zip(all_samples, all_adapt_params):
            for i in range(self._num_grad_updates):
                self._optimize(itr, *samples[i], set_grad=set_grad)

            update_module_params(self._old_policy, adapt_params)
            with torch.set_grad_enabled(set_grad):
                kl = self._inner_algo._compute_kl_constraint(samples[-1][2])
            kls.append(kl)

            zero_grad(theta.values())
            update_module_params(self.policy, theta)
            update_module_params(self._old_policy, old_theta)

        return torch.stack(kls).mean()

    def _compute_policy_entropy(self, task_samples):
        entropies = [
            self._inner_algo._compute_policy_entropy(samples[2])
                for samples in task_samples
        ]
        return torch.stack(entropies).mean()

    @property
    def _old_policy(self):
        return self._inner_algo._old_policy

    def _process_samples(self, itr, paths):
        for path in paths:
            path['returns'] = tensor_utils.discount_cumsum(
                path['rewards'], self._inner_algo.discount)
        self.baseline.fit(paths)
        return self._inner_algo.process_samples(itr, paths)

    def _log(self, itr, all_samples, loss_before, loss_after, kl_before,
                kl, policy_entropy):

        tabular.record('Iteration', itr)

        for i in range(self._num_grad_updates + 1):
            paths = [
                path for task_samples in all_samples for path in task_samples[i][0]
            ]

            average_discounted_return = (np.mean(
                [path['returns'][0] for path in paths]))
            undiscounted_returns = [sum(path['rewards']) for path in paths]
            average_return = np.mean(undiscounted_returns)
            self._episode_reward_mean[i].extend(undiscounted_returns)

            with tabular.prefix("Update_{0}/".format(i)):
                tabular.record('AverageDiscountedReturn', average_discounted_return)
                tabular.record('AverageReturn', average_return)
                tabular.record('Extras/EpisodeRewardMean',
                            np.mean(self._episode_reward_mean[i]))
                tabular.record('StdReturn', np.std(undiscounted_returns))
                tabular.record('MaxReturn', np.max(undiscounted_returns))
                tabular.record('MinReturn', np.min(undiscounted_returns))
                tabular.record('NumTrajs', len(paths))

        with tabular.prefix(self.policy.name + "/"):
            tabular.record('LossBefore', loss_before)
            tabular.record('LossAfter', loss_after)
            tabular.record('dLoss', loss_before - loss_after)
            tabular.record('KLBefore', kl_before)
            tabular.record('KLAfter', kl)
            tabular.record('Entropy', policy_entropy)

        return average_return
