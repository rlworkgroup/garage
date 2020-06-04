"""Module for RL2.

This module contains RL2, RL2Worker and the environment wrapper for RL2.
"""
import abc
import collections

import akro
from dowel import logger
import gym
import numpy as np

from garage import log_multitask_performance, TrajectoryBatch
from garage.envs import EnvSpec
from garage.misc import tensor_utils as np_tensor_utils
from garage.np.algos import MetaRLAlgorithm
from garage.sampler import DefaultWorker
from garage.tf.algos._rl2npo import RL2NPO


class RL2Env(gym.Wrapper):
    """Environment wrapper for RL2.

    In RL2, observation is concatenated with previous action,
    reward and terminal signal to form new observation.

    Args:
        env (gym.Env): An env that will be wrapped.

    """

    def __init__(self, env):
        super().__init__(env)
        action_space = akro.from_gym(self.env.action_space)
        observation_space = self._create_rl2_obs_space()
        self._spec = EnvSpec(action_space=action_space,
                             observation_space=observation_space)

    def _create_rl2_obs_space(self):
        """Create observation space for RL2.

        Returns:
            gym.spaces.Box: Augmented observation space.

        """
        obs_flat_dim = np.prod(self.env.observation_space.shape)
        action_flat_dim = np.prod(self.env.action_space.shape)
        return akro.Box(low=-np.inf,
                        high=np.inf,
                        shape=(obs_flat_dim + action_flat_dim + 1 + 1, ))

    def reset(self, **kwargs):
        """gym.Env reset function.

        Args:
            kwargs: Keyword arguments.

        Returns:
            np.ndarray: augmented observation.

        """
        del kwargs
        obs = self.env.reset()
        return np.concatenate(
            [obs, np.zeros(self.env.action_space.shape), [0], [0]])

    def step(self, action):
        """gym.Env step function.

        Args:
            action (int): action taken.

        Returns:
            np.ndarray: augmented observation.
            float: reward.
            bool: terminal signal.
            dict: environment info.

        """
        next_obs, reward, done, info = self.env.step(action)
        next_obs = np.concatenate([next_obs, action, [reward], [done]])
        return next_obs, reward, done, info

    @property
    def spec(self):
        """Environment specification.

        Returns:
            EnvSpec: Environment specification.

        """
        return self._spec


class RL2Worker(DefaultWorker):
    """Initialize a worker for RL2.

    In RL2, policy does not reset between trajectories in each meta batch.
    Policy only resets once at the beginning of a trial/meta batch.

    Args:
        seed(int): The seed to use to intialize random number generators.
        max_path_length(int or float): The maximum length paths which will
            be sampled. Can be (floating point) infinity.
        worker_number(int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        n_paths_per_trial (int): Number of trajectories sampled per trial/
            meta batch. Policy resets in the beginning of a meta batch,
            and obtain `n_paths_per_trial` trajectories in one meta batch.

    Attributes:
        agent(Policy or None): The worker's agent.
        env(gym.Env or None): The worker's environment.

    """

    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_path_length,
            worker_number,
            n_paths_per_trial=2):
        self._n_paths_per_trial = n_paths_per_trial
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)

    def start_rollout(self):
        """Begin a new rollout."""
        self._path_length = 0
        self._prev_obs = self.env.reset()

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        self.agent.reset()
        for _ in range(self._n_paths_per_trial):
            self.start_rollout()
            while not self.step_rollout():
                pass
        self._agent_infos['batch_idx'] = np.full(len(self._rewards),
                                                 self._worker_number)
        return self.collect_rollout()


class NoResetPolicy:
    """A policy that does not reset.

    For RL2 meta-test, the policy should not reset after meta-RL
    adapation. The hidden state will be retained as it is where
    the adaptation takes place.

    Args:
        policy (garage.tf.policies.Policy): Policy itself.

    Returns:
        object: The wrapped policy that does not reset.

    """

    def __init__(self, policy):
        self._policy = policy

    def reset(self):
        """gym.Env reset function."""

    def get_action(self, obs):
        """Get a single action from this policy for the input observation.

        Args:
            obs (numpy.ndarray): Observation from environment.

        Returns:
            tuple[numpy.ndarray, dict]: Predicted action and agent
                info.

        """
        return self._policy.get_action(obs)

    def get_param_values(self):
        """Return values of params.

        Returns:
            np.ndarray: Policy parameters values.

        """
        return self._policy.get_param_values()

    def set_param_values(self, params):
        """Set param values.

        Args:
            params (np.ndarray): A numpy array of parameter values.

        """
        self._policy.set_param_values(params)


# pylint: disable=protected-access
class RL2AdaptedPolicy:
    """A RL2 policy after adaptation.

    Args:
        policy (garage.tf.policies.Policy): Policy itself.

    """

    def __init__(self, policy):
        self._initial_hiddens = policy._prev_hiddens[:]
        self._policy = policy

    def reset(self):
        """gym.Env reset function."""
        self._policy._prev_hiddens = self._initial_hiddens

    def get_action(self, obs):
        """Get a single action from this policy for the input observation.

        Args:
            obs (numpy.ndarray): Observation from environment.

        Returns:
            tuple(numpy.ndarray, dict): Predicted action and agent info.

        """
        return self._policy.get_action(obs)

    def get_param_values(self):
        """Return values of params.

        Returns:
            tuple(np.ndarray, np.ndarray): Policy parameters values
                and initial hidden state that will be set every time
                the policy is used for meta-test.

        """
        return (self._policy.get_param_values(), self._initial_hiddens)

    def set_param_values(self, params):
        """Set param values.

        Args:
            params (tuple(np.ndarray, np.ndarray)): Two numpy array of
                parameter values, one of the network parameters, one
                for the initial hidden state.

        """
        inner_params, hiddens = params
        self._policy.set_param_values(inner_params)
        self._initial_hiddens = hiddens


class RL2(MetaRLAlgorithm, abc.ABC):
    """RL^2.

    Reference: https://arxiv.org/pdf/1611.02779.pdf.

    When sampling for RL^2, there are more than one environments to be
    sampled from. In the original implementation, within each task/environment,
    all rollouts sampled will be concatenated into one single rollout, and fed
    to the inner algorithm. Thus, returns and advantages are calculated across
    the rollout.

    RL2Worker is required in sampling for RL2.
    See example/tf/rl2_ppo_halfcheetah.py for reference.

    User should not instantiate RL2 directly.
    Currently garage supports PPO and TRPO as inner algorithm. Refer to
    garage/tf/algos/rl2ppo.py and garage/tf/algos/rl2trpo.py.

    Args:
        rl2_max_path_length (int): Maximum length for trajectories with respect
            to RL^2. Notice that it is different from the maximum path length
            for the inner algorithm.
        meta_batch_size (int): Meta batch size.
        task_sampler (garage.experiment.TaskSampler): Task sampler.
        meta_evaluator (garage.experiment.MetaEvaluator): Evaluator for meta-RL
            algorithms.
        n_epochs_per_eval (int): If meta_evaluator is passed, meta-evaluation
            will be performed every `n_epochs_per_eval` epochs.
        inner_algo_args (dict): Arguments for inner algorithm.

    """

    def __init__(self, rl2_max_path_length, meta_batch_size, task_sampler,
                 meta_evaluator, n_epochs_per_eval, **inner_algo_args):
        self._inner_algo = RL2NPO(**inner_algo_args)
        self._rl2_max_path_length = rl2_max_path_length
        self.env_spec = self._inner_algo._env_spec
        self._n_epochs_per_eval = n_epochs_per_eval
        self._flatten_input = self._inner_algo._flatten_input
        self._policy = self._inner_algo.policy
        self._discount = self._inner_algo._discount
        self._meta_batch_size = meta_batch_size
        self._task_sampler = task_sampler
        self._meta_evaluator = meta_evaluator

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
            if runner.step_itr % self._n_epochs_per_eval == 0:
                if self._meta_evaluator is not None:
                    self._meta_evaluator.evaluate(self)
            runner.step_path = runner.obtain_samples(
                runner.step_itr,
                env_update=self._task_sampler.sample(self._meta_batch_size))
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
        self._inner_algo.optimize_policy(paths)
        return paths['average_return']

    def get_exploration_policy(self):
        """Return a policy used before adaptation to a specific task.

        Each time it is retrieved, this policy should only be evaluated in one
        task.

        Returns:
            object: The policy used to obtain samples that are later
                used for meta-RL adaptation.

        """
        self._policy.reset()
        return NoResetPolicy(self._policy)

    # pylint: disable=protected-access
    def adapt_policy(self, exploration_policy, exploration_trajectories):
        """Produce a policy adapted for a task.

        Args:
            exploration_policy (garage.Policy): A policy which was returned
                from get_exploration_policy(), and which generated
                exploration_trajectories by interacting with an environment.
                The caller may not use this object after passing it into this
                method.
            exploration_trajectories (garage.TrajectoryBatch): Trajectories to
                adapt to, generated by exploration_policy exploring the
                environment.

        Returns:
            garage.tf.policies.Policy: A policy adapted to the task represented
                by the exploration_trajectories.

        """
        return RL2AdaptedPolicy(exploration_policy._policy)

    # pylint: disable=protected-access
    def _process_samples(self, itr, paths):
        # pylint: disable=too-many-statements
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (OrderedDict[dict]): A list of collected paths for each
                task. In RL^2, there are n environments/tasks and paths in
                each of them will be concatenated at some point and fed to
                the policy.

        Returns:
            dict: Processed sample data, with key
                * observations: (numpy.ndarray)
                * actions: (numpy.ndarray)
                * rewards: (numpy.ndarray)
                * returns: (numpy.ndarray)
                * valids: (numpy.ndarray)
                * agent_infos: (dict)
                * env_infos: (dict)
                * paths: (list[dict])
                * average_return: (numpy.float64)

        Raises:
            ValueError: If 'batch_idx' is not found.

        """
        concatenated_paths = []

        paths_by_task = collections.defaultdict(list)
        for path in paths:
            path['returns'] = np_tensor_utils.discount_cumsum(
                path['rewards'], self._discount)
            path['lengths'] = [len(path['rewards'])]
            if 'batch_idx' in path:
                paths_by_task[path['batch_idx']].append(path)
            elif 'batch_idx' in path['agent_infos']:
                paths_by_task[path['agent_infos']['batch_idx'][0]].append(path)
            else:
                raise ValueError(
                    'Batch idx is required for RL2 but not found, '
                    'Make sure to use garage.tf.algos.rl2.RL2Worker '
                    'for sampling')

        # all path in paths_by_task[i] are sampled from task[i]
        for _paths in paths_by_task.values():
            concatenated_path = self._concatenate_paths(_paths)
            concatenated_paths.append(concatenated_path)

        # stack and pad to max path length of the concatenated
        # path, which will be fed to inner algo
        # i.e. max_path_length * episode_per_task
        concatenated_paths_stacked = (
            np_tensor_utils.stack_and_pad_tensor_dict_list(
                concatenated_paths, self._inner_algo.max_path_length))

        name_map = None
        if hasattr(self._task_sampler, '_envs') and hasattr(
                self._task_sampler._envs[0].env, 'all_task_names'):
            names = [
                env.env.all_task_names[0] for env in self._task_sampler._envs
            ]
            name_map = dict(enumerate(names))

        undiscounted_returns = log_multitask_performance(
            itr,
            TrajectoryBatch.from_trajectory_list(self.env_spec, paths),
            self._inner_algo._discount,
            name_map=name_map)

        concatenated_paths_stacked['paths'] = concatenated_paths
        concatenated_paths_stacked['average_return'] = np.mean(
            undiscounted_returns)

        return concatenated_paths_stacked

    def _concatenate_paths(self, paths):
        """Concatenate paths.

        The input paths are from different rollouts but same task/environment.
        In RL^2, paths within each meta batch are all concatenate into a single
        path and fed to the policy.

        Args:
            paths (dict): Input paths. All paths are from different rollouts,
                but the same task/environment.

        Returns:
            dict: Concatenated paths from the same task/environment. Shape of
                values: :math:`[max_path_length * episode_per_task, S^*]`
            list[dict]: Original input paths. Length of the list is
                :math:`episode_per_task` and each path in the list has
                values of shape :math:`[max_path_length, S^*]`

        """
        if self._flatten_input:
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
        valids = np.concatenate(
            [np.ones_like(path['rewards']) for path in paths])
        baselines = np.concatenate(
            [np.zeros_like(path['rewards']) for path in paths])

        concatenated_path = np_tensor_utils.concat_tensor_dict_list(paths)
        concatenated_path['observations'] = observations
        concatenated_path['actions'] = actions
        concatenated_path['valids'] = valids
        concatenated_path['baselines'] = baselines

        return concatenated_path

    @property
    def policy(self):
        """Policy.

        Returns:
            garage.Policy: Policy to be used.

        """
        return self._inner_algo.policy

    @property
    def max_path_length(self):
        """Max path length.

        Returns:
            int: Maximum path length in a trajectory.

        """
        return self._rl2_max_path_length
