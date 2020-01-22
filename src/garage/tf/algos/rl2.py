"""RL^2: Fast Reinforcement learning via slow reinforcement learning.

Reference: https://arxiv.org/pdf/1611.02779.pdf.
"""
import collections
import copy

from dowel import tabular
import numpy as np

from garage.misc import tensor_utils as np_tensor_utils
from garage.tf.algos import BatchPolopt


class RL2(BatchPolopt):
    # pylint: disable=abstract-method
    """RL^2 .

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_path_length (int): Maximum length of a single rollout.
        episode_per_task (int): Number of episode to be sampled per task.
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
        flatten_input (bool): Whether to flatten input along the observation
            dimension. If True, for example, an observation with shape (2, 4)
            will be flattened to 8.
        num_of_env (int): Number of vectorized environment instances to be
            used for sampling.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_path_length=500,
                 episode_per_task=1,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 flatten_input=True,
                 num_of_env=1):
        self._episode_per_task = episode_per_task
        self._rl2_max_path_length = max_path_length // episode_per_task
        self._num_of_env = num_of_env
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

        self.baselines = [
            copy.deepcopy(self.baseline) for i in range(num_of_env)
        ]

    @property
    def rl2_max_path_length(self):
        """Max path length for RL^2.

        This is different from max_path_length, which represents maximum
        path length for concatenated paths. This represents the maximum
        path length for each individual paths. Usually the value would
        be maximum_path_length * episode_per_task.

        Returns:
            int: Maximum path length for RL^2.

        """
        return self._rl2_max_path_length

    def process_samples(self, itr, paths):
        # pylint: disable=too-many-statements
        """Return processed sample data based on the collected paths.

        In RL^2, paths in each of meta batch will be concatenated and
        fed to the policy.
        This function process the individual paths from all rollouts in
        each environment/task by doing the followings:

        * Calculate cumulative returns for each step in each individual path
        * Concatenate all paths from each meta batch into one single path.
        * Stack all concatenated paths

        Args:
            itr (int): Iteration number.
            paths (list[Dict]): A list of collected paths for each task.

        Returns:
            dict: Processed sample data, with values:
                numpy.ndarray: Observations. Shape:
                    :math:`[N, episode_per_task * max_path_length, S^*]`
                numpy.ndarray: Actions. Shape:
                    :math:`[N, episode_per_task * max_path_length, S^*]`
                numpy.ndarray: Rewards. Shape:
                    :math:`[N, episode_per_task * max_path_length, S^*]`
                numpy.ndarray: Terminal signals. Shape:
                    :math:`[N, episode_per_task * max_path_length, S^*]`
                numpy.ndarray: Returns. Shape:
                    :math:`[N, episode_per_task * max_path_length, S^*]`
                numpy.ndarray: Valids. Shape:
                    :math:`[N, episode_per_task * max_path_length, S^*]`
                numpy.ndarray: Lengths. Shape:
                    :math:`[N, episode_per_task]`
                dict[numpy.ndarray]: Environment Infos. Shape of values:
                    :math:`[N, episode_per_task * max_path_length, S^*]`
                dict[numpy.ndarray]: Agent Infos. Shape of values:
                    :math:`[N, episode_per_task * max_path_length, S^*]`

        """
        concatenated_path_in_meta_batch = []
        lengths = []

        paths_by_task = collections.defaultdict(list)
        for path in paths:
            path['returns'] = np_tensor_utils.discount_cumsum(
                path['rewards'], self.discount)
            path['lengths'] = len(path['rewards'])
            batch_id = path['batch_idx']
            path['baselines'] = self.baselines[batch_id].predict(path)
            paths_by_task[batch_id].append(path)

        for path in paths_by_task.values():
            concatenated_path = self._concatenate_paths(path)
            concatenated_path_in_meta_batch.append(concatenated_path)

        (observations, actions, rewards, _, _, valids, baselines, lengths,
         env_infos, agent_infos) = \
            self._stack_paths(
                max_len=self.max_path_length,
                paths=concatenated_path_in_meta_batch)

        (_observations, _actions, _rewards, _terminals, _, _valids, _,
         _lengths, _env_infos, _agent_infos) = \
            self._stack_paths(
                max_len=self._rl2_max_path_length,
                paths=paths)

        ent = np.sum(self.policy.distribution.entropy(agent_infos) *
                     valids) / np.sum(valids)

        # performance is evaluated across all paths
        undiscounted_returns = self.evaluate_performance(
            itr,
            dict(env_spec=self.env_spec,
                 observations=_observations,
                 actions=_actions,
                 rewards=_rewards,
                 terminals=_terminals,
                 env_infos=_env_infos,
                 agent_infos=_agent_infos,
                 lengths=_lengths,
                 discount=self.discount))

        tabular.record('Entropy', ent)
        tabular.record('Perplexity', np.exp(ent))

        # all paths in each meta batch is stacked together
        # shape: [meta_batch, max_path_length * episoder_per_task, *dims]
        # per RL^2
        concatenated_path = dict(observations=observations,
                                 actions=actions,
                                 rewards=rewards,
                                 valids=valids,
                                 lengths=lengths,
                                 baselines=baselines,
                                 agent_infos=agent_infos,
                                 env_infos=env_infos,
                                 paths=concatenated_path_in_meta_batch,
                                 average_return=np.mean(undiscounted_returns))

        return concatenated_path

    def _concatenate_paths(self, paths):
        """Concatenate paths.

        The input paths are from different rollouts but same task/environment.
        In RL^2, paths within each meta batch are all concatenated into a
        single path and fed to the policy.

        Args:
            paths (dict): Input paths. All paths are from different rollouts,
                but the same task/environment.

        Returns:
            dict: Concatenated paths from the same task/environment. Shape of
                values: :math:`[max_path_length * episode_per_task, S^*]`
            list[dict]: Original input paths. Length of the list is
                :math:`episode_per_task` and each path in the list has values
                of shape :math:`[max_path_length, S^*]`

        """
        returns = []

        if self.flatten_input:
            observations = np.concatenate([
                self.env_spec.observation_space.flatten_n(path['observations'])
                for path in paths
            ])
        else:
            observations = np.concatenate(
                [path['observations'] for path in paths])
        actions = np.concatenate([
            self.env_spec.action_space.flatten_n(path['actions'])
            for path in paths
        ])
        rewards = np.concatenate([path['rewards'] for path in paths])
        dones = np.concatenate([path['dones'] for path in paths])
        valids = np.concatenate(
            [np.ones_like(path['rewards']) for path in paths])
        returns = np.concatenate([path['returns'] for path in paths])
        baselines = np.concatenate([path['baselines'] for path in paths])
        env_infos = np_tensor_utils.concat_tensor_dict_list(
            [path['env_infos'] for path in paths])
        agent_infos = np_tensor_utils.concat_tensor_dict_list(
            [path['agent_infos'] for path in paths])
        lengths = [path['lengths'] for path in paths]

        concatenated_path = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            valids=valids,
            baselines=baselines,
            lengths=lengths,
            returns=returns,
            agent_infos=agent_infos,
            env_infos=env_infos,
        )
        return concatenated_path

    def _stack_paths(self, max_len, paths):
        # pylint: disable=no-self-use
        """Pad paths to max_len and stacked all paths together.

        Args:
            max_len (int): Maximum path length.
            paths (dict): Input paths. Each path represents the concatenated
                paths from each meta batch (environment/task).

        Returns:
            numpy.ndarray: Observations. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Actions. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Rewards. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Terminal signals. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Returns. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Valids. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Lengths. Shape:
                :math:`[meta_batch, episode_per_task]`
            dict[numpy.ndarray]: Environment Infos. Shape of values:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            dict[numpy.ndarray]: Agent Infos. Shape of values:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`

        """
        observations = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'observations', max_len)
        actions = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'actions', max_len)
        rewards = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'rewards', max_len)
        dones = np_tensor_utils.stack_and_pad_tensor_n(paths, 'dones', max_len)
        returns = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'returns', max_len)
        baselines = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'baselines', max_len)

        agent_infos = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'agent_infos', max_len)
        env_infos = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'env_infos', max_len)

        valids = [np.ones_like(path['rewards']) for path in paths]
        valids = np_tensor_utils.pad_tensor_n(valids, max_len)

        lengths = np.stack([path['lengths'] for path in paths])

        return (observations, actions, rewards, dones, returns, valids,
                baselines, lengths, env_infos, agent_infos)
