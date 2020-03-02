# pylint: disable=unnecessary-pass
"""A sampler for PEARL."""
import numpy as np

from garage.misc import tensor_utils


class PEARLSampler:
    """A sampler used in meta-RL algorithms involving context.

    It stores context and resample belief in the policy every step.

    Args:
        env (garage.envs.GarageEnv): An environement instance.
        policy (garage.policies.Policy): Policy used for sampling.
        max_path_length (int): Maximum length of path for each step

    """

    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length

    def start_worker(self):
        """Start worker."""
        pass

    def shutdown_worker(self):
        """Terminate worker."""
        pass

    def obtain_samples(self,
                       max_samples=np.inf,
                       max_trajs=np.inf,
                       deterministic=False,
                       accum_context=True,
                       resample_rate=1):
        """Obtain samples in the environment up to max_samples or max_trajs.

        Args:
            max_samples (int): Maximum number of samples.
            max_trajs (int):  Maximum number of trajectories.
            deterministic (bool): Whether or not policy is deterministic.
            accum_context (bool): Whether or not to update the collected
                context in policy.
            resample_rate (int): How often (in trajectories) to resample
                context.

        Returns:
            list: A list of paths.
            int: Total number of steps.

        """
        assert max_samples < np.inf or max_trajs < np.inf
        paths = []
        n_steps_total = 0
        n_trajs = 0

        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = self.rollout(max_path_length=self.max_path_length,
                                deterministic=deterministic,
                                accum_context=accum_context)

            # save the latent context that generated this trajectory
            path['context'] = self.policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1

            # resample z every transition
            if n_trajs % resample_rate == 0:
                self.policy.sample_from_belief()
        return paths, n_steps_total

    def rollout(self,
                max_path_length=np.inf,
                deterministic=False,
                accum_context=True):
        """Sample a single rollout of the agent in the environment.

        Args:
            max_path_length(int): If the rollout reaches this many timesteps,
                it is terminated.
            deterministic (bool): If true, use the mean action returned by the
                stochastic policy instead of sampling from the returned action
                distribution.
            accum_context (bool): If true, update context of the agent.

        Returns:
            dict[str, np.ndarray or dict]: Dictionary, with keys:
                * observations(np.array): Non-flattened array of observations.
                    There should be one more of these than actions. Should have
                    shape (T, S^*) (the unflattened state space of the current
                    environment).
                * actions(np.array): Non-flattened array of actions. Should have
                    shape (T, S^*).
                * rewards(np.array): Array of rewards of shape (T,) (1D array of
                    length timesteps).
                * next_observations(np.array):  Non-flattened array of next
                    observations. Should have shape (T, S^*).
                * terminals(np.array): Array of terminals of shape (T, 1).
                * agent_infos(Dict[str, np.array]): Dictionary of stacked,
                    non-flattened `agent_info` arrays.
                * env_infos(Dict[str, np.array]): Dictionary of stacked,
                    non-flattened `env_info` arrays.

        """
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        o = self.env.reset()
        next_o = None
        self.policy.reset()
        path_length = 0
        while path_length < max_path_length:
            a, agent_info = self.policy.get_action(o)
            if deterministic:
                a = agent_info['mean']
            next_o, r, d, env_info = self.env.step(a)
            if accum_context:
                self.policy.update_context([o, a, r, next_o, d, env_info])

            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            o = next_o

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            next_o = np.array([next_o])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )

        return dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        )
