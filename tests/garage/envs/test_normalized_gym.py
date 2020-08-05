from garage.envs import GymEnv, normalize


class TestNormalizedGym:

    def setup_method(self):
        self.env = normalize(GymEnv('Pendulum-v0'),
                             normalize_reward=True,
                             normalize_obs=True,
                             flatten_obs=True)

    def teardown_method(self):
        self.env.close()

    def test_does_not_modify_action(self):
        a = self.env.action_space.sample()
        a_copy = a
        self.env.reset()
        self.env.step(a)
        assert a == a_copy

    def test_flatten(self):
        for _ in range(10):
            self.env.reset()
            self.env.visualize()
            for _ in range(5):
                action = self.env.action_space.sample()
                es = self.env.step(action)
                next_obs, done = es.observation, es.terminal
                assert next_obs.shape == self.env.observation_space.low.shape
                if done:
                    break

    def test_unflatten(self):
        for _ in range(10):
            self.env.reset()
            for _ in range(5):
                action = self.env.action_space.sample()
                es = self.env.step(action)
                next_obs, done = es.observation, es.terminal
                # yapf: disable
                assert (self.env.observation_space.flatten(next_obs).shape
                        == self.env.observation_space.flat_dim)
                # yapf: enable
                if done:
                    break
