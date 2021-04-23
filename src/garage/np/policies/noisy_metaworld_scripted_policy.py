from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
import metaworld.policies
import numpy as np

from garage.np.policies import Policy


class NoisyMetaworldScriptedPolicy(Policy):

    def __init__(self, env_spec, env_name, act_noise_pct):
        name_caps_case = env_name.title().replace('-', '')
        policy_class = getattr(metaworld.policies,
                               f'Sawyer{name_caps_case}V2Policy', None)
        if policy_class is None:
            if name_caps_case == 'PegInsertSide':
                name_caps_case = 'PegInsertionSide'
            policy_class = getattr(metaworld.policies,
                                   f'Sawyer{name_caps_case}V2Policy', None)
            if policy_class is None:
                raise ValueError(f'Policy missing for {env_name}')
        self._policy = policy_class()
        self.noise = act_noise_pct
        self._env_spec = env_spec
        self._name = f'NoisyMetaworldScriptedPolicy({env_name})'

    @property
    def name(self):
        return self._name

    @property
    def env_spec(self):
        return self._env_spec

    def get_action(self, o):
        action_space_ptp = self._env_spec.action_space.high - self._env_spec.action_space.low
        original_a = self._policy.get_action(o)
        noisy_a = np.random.normal(original_a, self.noise * action_space_ptp)
        return noisy_a, {'action_targets': original_a}

    def get_actions(self, observations):
        """Get actions from this policy for the input observation.

        Args:
            observations(list): Observations from the environment.

        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).

        """
        actions = []
        infos = defaultdict(list)
        for obs in observations:
            act, info = self.get_action(obs)
            actions.append(act)
            for (k, v) in info.items():
                infos[k].append(v)
        return np.array(actions), {k: np.array(v) for (k, v) in infos.items()}
