"""RL^2: Fast Reinforcement learning via slow reinforcement learning.

Reference: https://arxiv.org/pdf/1611.02779.pdf.
"""
import collections

from dowel import logger, tabular
import numpy as np

import garage
from garage.misc import tensor_utils as np_tensor_utils
from garage.np.algos import RLAlgorithm


class RL2(RLAlgorithm):
    """RL^2 .

    Args:
        inner_algo (garage.tf.algos.BatchPolopt): Inner algorithm. RL^2 only
            works with garage.tf.algos.BatchPolopt.
        max_path_length (int): Maximum length for trajectories with respect
            to RL^2. Notice that it is differen from the maximum path length
            for the inner algorithm.

    Raises:
        ValueError: If the inner algorithm is not of type
            garage.tf.algos.BatchPolopt.

    """

    def __init__(self, inner_algo, max_path_length):
        if not isinstance(inner_algo, garage.tf.algos.BatchPolopt):
            raise ValueError('RL^2 only works with '
                             'garage.tf.algos.BatchPolopt!')
        self._inner_algo = inner_algo
        self._max_path_length = max_path_length
        self._env_spec = inner_algo.env_spec
        self._flatten_input = inner_algo.flatten_input
        self._policy = inner_algo.policy
        self._discount = inner_algo.discount

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch.

        """
        last_return = None

        for _ in runner.step_epochs():
            runner.step_path = runner.obtain_samples(runner.step_itr)
            tabular.record('TotalEnvSteps', runner.total_env_steps)
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
        paths = self._process_samples(itr, paths)
        logger.log('Optimizing policy...')
        self._inner_algo.optimize_policy(itr, paths)
        return paths['average_return']

    def _process_samples(self, itr, paths):
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
                path['rewards'], self._discount)
            path['lengths'] = len(path['rewards'])
            paths_by_task[path['batch_idx']].append(path)

        for path in paths_by_task.values():
            concatenated_path = self._concatenate_paths(path)
            concatenated_path_in_meta_batch.append(concatenated_path)

        (observations, actions, rewards, _, _, valids, lengths,
         env_infos, agent_infos) = \
            self._stack_paths(
                max_len=self._inner_algo.max_path_length,
                paths=concatenated_path_in_meta_batch)

        (_observations, _actions, _rewards, _terminals, _, _valids, _lengths,
         _env_infos, _agent_infos) = \
            self._stack_paths(
                max_len=self._max_path_length,
                paths=paths)

        ent = np.sum(self._policy.distribution.entropy(agent_infos) *
                     valids) / np.sum(valids)

        # performance is evaluated across all paths
        undiscounted_returns = self.evaluate_performance(
            itr,
            dict(env_spec=self._env_spec,
                 observations=_observations,
                 actions=_actions,
                 rewards=_rewards,
                 terminals=_terminals,
                 env_infos=_env_infos,
                 agent_infos=_agent_infos,
                 lengths=_lengths,
                 discount=self._discount))

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

        if self._flatten_input:
            observations = np.concatenate([
                self._env_spec.observation_space.flatten_n(
                    path['observations']) for path in paths
            ])
        else:
            observations = np.concatenate(
                [path['observations'] for path in paths])
        actions = np.concatenate([
            self._env_spec.action_space.flatten_n(path['actions'])
            for path in paths
        ])
        rewards = np.concatenate([path['rewards'] for path in paths])
        dones = np.concatenate([path['dones'] for path in paths])
        valids = np.concatenate(
            [np.ones_like(path['rewards']) for path in paths])
        returns = np.concatenate([path['returns'] for path in paths])

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

        agent_infos = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'agent_infos', max_len)
        env_infos = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'env_infos', max_len)

        valids = [np.ones_like(path['rewards']) for path in paths]
        valids = np_tensor_utils.pad_tensor_n(valids, max_len)

        lengths = np.stack([path['lengths'] for path in paths])

        return (observations, actions, rewards, dones, returns, valids,
                lengths, env_infos, agent_infos)

    @property
    def policy(self):
        """Policy used for inner algorithm.

        Returns:
            garage.tf.policies.base.Policy: Policy used for inner algorithm.

        """
        return self._policy

    @property
    def max_path_length(self):
        """Maximum path length.

        Returns:
            int: Maximum path length.

        """
        return self._max_path_length
