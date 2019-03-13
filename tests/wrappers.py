import gym


class AutoStopEnv(gym.Wrapper):
    """A env wrapper that stops rollout at step max_path_length."""

    def __init__(self, env=None, env_name="", max_path_length=100):
        if env_name:
            super().__init__(gym.make(env_name))
        else:
            super().__init__(env)
        self._rollout_step = 0
        self._max_path_length = max_path_length

    def step(self, actions):
        self._rollout_step += 1
        next_obs, reward, done, info = self.env.step(actions)
        if self._rollout_step == self._max_path_length:
            done = True
            self._rollout_step = 0
        return next_obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
