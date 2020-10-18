"""Module for RL2.

This module contains RL2, RL2Worker and the environment wrapper for RL2.
"""
# yapf: disable
import abc
import collections

import akro
from dowel import logger
import numpy as np

from garage import (EnvSpec,
                    EnvStep,
                    EpisodeBatch,
                    log_multitask_performance,
                    StepType,
                    Wrapper)
from garage.np.algos import MetaRLAlgorithm
from garage.sampler import DefaultWorker
from garage.tf.algos._rl2npo import RL2NPO

# yapf: enable


class RL2Env(Wrapper):
    """Environment wrapper for RL2.

    In RL2, observation is concatenated with previous action,
    reward and terminal signal to form new observation.

    Args:
        env (Environment): An env that will be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)

        self._observation_space = self._create_rl2_obs_space()
        self._spec = EnvSpec(
            action_space=self.action_space,
            observation_space=self._observation_space,
            max_episode_length=self._env.spec.max_episode_length)

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    def reset(self):
        """Call reset on wrapped env.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """
        first_obs, episode_info = self._env.reset()
        first_obs = np.concatenate(
            [first_obs,
             np.zeros(self._env.action_space.shape), [0], [0]])

        return first_obs, episode_info

    def step(self, action):
        """Call step on wrapped env.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            EnvStep: The environment step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.

        """
        es = self._env.step(action)
        next_obs = es.observation
        next_obs = np.concatenate([
            next_obs, action, [es.reward], [es.step_type == StepType.TERMINAL]
        ])

        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=es.reward,
                       observation=next_obs,
                       env_info=es.env_info,
                       step_type=es.step_type)

    def _create_rl2_obs_space(self):
        """Create observation space for RL2.

        Returns:
            akro.Box: Augmented observation space.

        """
        obs_flat_dim = np.prod(self._env.observation_space.shape)
        action_flat_dim = np.prod(self._env.action_space.shape)
        return akro.Box(low=-np.inf,
                        high=np.inf,
                        shape=(obs_flat_dim + action_flat_dim + 1 + 1, ))


class RL2Worker(DefaultWorker):
    """Initialize a worker for RL2.

    In RL2, policy does not reset between epsiodes in each meta batch.
    Policy only resets once at the beginning of a trial/meta batch.

    Args:
        seed (int): The seed to use to intialize random number generators.
        max_episode_length (int or float): The maximum length of episodes to
            sample. Can be (floating point) infinity.
        worker_number (int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        n_episodes_per_trial (int): Number of episodes sampled per
            trial/meta-batch. Policy resets in the beginning of a meta batch,
            and obtain `n_episodes_per_trial` episodes in one meta batch.

    Attributes:
        agent(Policy or None): The worker's agent.
        env(Environment or None): The worker's environment.

    """

    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_episode_length,
            worker_number,
            n_episodes_per_trial=2):
        self._n_episodes_per_trial = n_episodes_per_trial
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)

    def start_episode(self):
        """Begin a new episode."""
        self._eps_length = 0
        self._prev_obs = self.env.reset()[0]

    def rollout(self):
        """Sample a single episode of the agent in the environment.

        Returns:
            EpisodeBatch: The collected episode.

        """
        self.agent.reset()
        for _ in range(self._n_episodes_per_trial):
            self.start_episode()
            while not self.step_episode():
                pass
        self._agent_infos['batch_idx'] = np.full(len(self._env_steps),
                                                 self._worker_number)
        return self.collect_episode()


class NoResetPolicy:
    """A policy that does not reset.

    For RL2 meta-test, the policy should not reset after meta-RL
    adapation. The hidden state will be retained as it is where
    the adaptation takes place.

    Args:
        policy (garage.tf.policies.Policy): Policy itself.

    Returns:
        garage.tf.policies.Policy: The wrapped policy that does not reset.

    """

    def __init__(self, policy):
        self._policy = policy

    def reset(self):
        """Environment reset function."""

    def get_action(self, obs):
        """Get a single action from this policy for the input observation.

        Args:
            obs (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Predicted action
            dict: Agent into

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
        """Environment reset function."""
        self._policy._prev_hiddens = self._initial_hiddens

    def get_action(self, obs):
        """Get a single action from this policy for the input observation.

        Args:
            obs (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Predicated action
            dict: Agent info.

        """
        return self._policy.get_action(obs)

    def get_param_values(self):
        """Return values of params.

        Returns:
            np.ndarray: Policy parameter values
            np.ndarray: Initial hidden state, which will be set every time
                the policy is used for meta-test.

        """
        return (self._policy.get_param_values(), self._initial_hiddens)

    def set_param_values(self, params):
        """Set param values.

        Args:
            params (Tuple[np.ndarray, np.ndarray]): Two numpy array of
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
    all episodes sampled will be concatenated into one single episode, and fed
    to the inner algorithm. Thus, returns and advantages are calculated across
    the episode.

    RL2Worker is required in sampling for RL2.
    See example/tf/rl2_ppo_halfcheetah.py for reference.

    User should not instantiate RL2 directly.
    Currently garage supports PPO and TRPO as inner algorithm. Refer to
    garage/tf/algos/rl2ppo.py and garage/tf/algos/rl2trpo.py.

    Args:
        env_spec (EnvSpec): Environment specification.
        episodes_per_trial (int): Used to calculate the max episode length for
            the inner algorithm.
        meta_batch_size (int): Meta batch size.
        task_sampler (TaskSampler): Task sampler.
        meta_evaluator (MetaEvaluator): Evaluator for meta-RL algorithms.
        n_epochs_per_eval (int): If meta_evaluator is passed, meta-evaluation
            will be performed every `n_epochs_per_eval` epochs.
        inner_algo_args (dict): Arguments for inner algorithm.

    """

    def __init__(self, env_spec, episodes_per_trial, meta_batch_size,
                 task_sampler, meta_evaluator, n_epochs_per_eval,
                 **inner_algo_args):
        self._env_spec = env_spec
        _inner_env_spec = EnvSpec(
            env_spec.observation_space, env_spec.action_space,
            episodes_per_trial * env_spec.max_episode_length)
        self._inner_algo = RL2NPO(env_spec=_inner_env_spec, **inner_algo_args)
        self._rl2_max_episode_length = self._env_spec.max_episode_length
        self._n_epochs_per_eval = n_epochs_per_eval
        self._policy = self._inner_algo.policy
        self._discount = self._inner_algo._discount
        self._meta_batch_size = meta_batch_size
        self._task_sampler = task_sampler
        self._meta_evaluator = meta_evaluator

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch.

        """
        last_return = None

        for _ in trainer.step_epochs():
            if trainer.step_itr % self._n_epochs_per_eval == 0:
                if self._meta_evaluator is not None:
                    self._meta_evaluator.evaluate(self)
            trainer.step_episode = trainer.obtain_episodes(
                trainer.step_itr,
                env_update=self._task_sampler.sample(self._meta_batch_size))
            last_return = self.train_once(trainer.step_itr,
                                          trainer.step_episode)
            trainer.step_itr += 1

        return last_return

    def train_once(self, itr, episodes):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.

        Returns:
            numpy.float64: Average return.

        """
        episodes, average_return = self._process_samples(itr, episodes)
        logger.log('Optimizing policy...')
        self._inner_algo.optimize_policy(episodes)
        return average_return

    def get_exploration_policy(self):
        """Return a policy used before adaptation to a specific task.

        Each time it is retrieved, this policy should only be evaluated in one
        task.

        Returns:
            Policy: The policy used to obtain samples that are later used for
                meta-RL adaptation.

        """
        self._policy.reset()
        return NoResetPolicy(self._policy)

    # pylint: disable=protected-access
    def adapt_policy(self, exploration_policy, exploration_episodes):
        """Produce a policy adapted for a task.

        Args:
            exploration_policy (Policy): A policy which was returned from
                get_exploration_policy(), and which generated
                exploration_episodes by interacting with an environment. The
                caller may not use this object after passing it into this
                method.
            exploration_episodes (EpisodeBatch): episodes to adapt to,
                generated by exploration_policy exploring the
                environment.

        Returns:
            Policy: A policy adapted to the task represented by the
                exploration_episodes.

        """
        return RL2AdaptedPolicy(exploration_policy._policy)

    # pylint: disable=protected-access
    def _process_samples(self, itr, episodes):
        # pylint: disable=too-many-statements
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Original collected episode batch for each
                task. For each episode, episode.agent_infos['batch_idx']
                indicates which task this episode belongs to. In RL^2, there
                are n environments/tasks and paths in each of them will be
                concatenated at some point and fed to the policy.

        Returns:
            EpisodeBatch: Processed batch of episodes for feeding the inner
                algorithm.
            numpy.float64: The average return.

        Raises:
            ValueError: If 'batch_idx' is not found.

        """
        concatenated_paths = []

        paths_by_task = collections.defaultdict(list)
        for episode in episodes.split():
            if hasattr(episode, 'batch_idx'):
                paths_by_task[episode.batch_idx[0]].append(episode)
            elif 'batch_idx' in episode.agent_infos:
                paths_by_task[episode.agent_infos['batch_idx'][0]].append(
                    episode)
            else:
                raise ValueError(
                    'Batch idx is required for RL2 but not found, '
                    'Make sure to use garage.tf.algos.rl2.RL2Worker '
                    'for sampling')

        # all path in paths_by_task[i] are sampled from task[i]
        for episode_list in paths_by_task.values():
            concatenated_path = self._concatenate_paths(episode_list)
            concatenated_paths.append(concatenated_path)

        concatenated_episodes = EpisodeBatch.concatenate(*concatenated_paths)

        name_map = None
        if hasattr(self._task_sampler, '_envs') and hasattr(
                self._task_sampler._envs[0]._env, 'all_task_names'):
            names = [
                env._env.all_task_names[0] for env in self._task_sampler._envs
            ]
            name_map = dict(enumerate(names))

        undiscounted_returns = log_multitask_performance(
            itr, episodes, self._inner_algo._discount, name_map=name_map)

        average_return = np.mean(undiscounted_returns)

        return concatenated_episodes, average_return

    def _concatenate_paths(self, episode_list):
        """Concatenate paths.

        The input paths are from different episodes but same task/environment.
        In RL^2, paths within each meta batch are all concatenate into a single
        path and fed to the policy.

        Args:
            episode_list (list[EpisodeBatch]): Input paths. All paths are from
                different episodes, but the same task/environment.

        Returns:
            dict: Concatenated paths from the same task/environment. Shape of
                values: :math:`[max_episode_length * episode_per_task, S^*]`
            list[dict]: Original input paths. Length of the list is
                :math:`episode_per_task` and each path in the list has
                values of shape :math:`[max_episode_length, S^*]`

        """
        env_infos = {
            k: np.concatenate([b.env_infos[k] for b in episode_list])
            for k in episode_list[0].env_infos.keys()
        }
        agent_infos = {
            k: np.concatenate([b.agent_infos[k] for b in episode_list])
            for k in episode_list[0].agent_infos.keys()
        }
        actions = np.concatenate([
            self._env_spec.action_space.flatten_n(ep.actions)
            for ep in episode_list
        ])

        return EpisodeBatch(
            env_spec=episode_list[0].env_spec,
            observations=np.concatenate(
                [ep.observations for ep in episode_list]),
            last_observations=episode_list[-1].last_observations,
            actions=actions,
            rewards=np.concatenate([ep.rewards for ep in episode_list]),
            env_infos=env_infos,
            agent_infos=agent_infos,
            step_types=np.concatenate([ep.step_types for ep in episode_list]),
            lengths=np.asarray([sum([ep.lengths[0] for ep in episode_list])]))

    @property
    def policy(self):
        """Policy: Policy to be used."""
        return self._inner_algo.policy

    @property
    def max_episode_length(self):
        """int: Maximum length of an episode."""
        return self._rl2_max_episode_length
