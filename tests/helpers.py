import pickle
import unittest

import tensorflow as tf

from tests.quirks import KNOWN_GYM_CLOSE_BROKEN
from tests.quirks import KNOWN_GYM_RENDER_NOT_IMPLEMENTED


class TfTest(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.sess.__enter__()

    def tearDown(self):
        self.sess.__exit__(None, None, None)
        self.sess.close()


def step_env(env, n=10, render=True):
    env.reset()
    for _ in range(n):
        _, _, done, _ = env.step(env.action_space.sample())
        if render:
            env.render()
        if done:
            break
    env.close()


def step_env_with_gym_quirks(test_case,
                             env,
                             spec,
                             n=10,
                             render=True,
                             serialize_env=False):
    if serialize_env:
        # Roundtrip serialization
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip.env.spec == env.env.spec
        env = round_trip

    env.reset()
    for _ in range(n):
        _, _, done, _ = env.step(env.action_space.sample())
        if render:
            if not spec.id in KNOWN_GYM_RENDER_NOT_IMPLEMENTED:
                env.render()
            else:
                with test_case.assertRaises(NotImplementedError):
                    env.render()
        if done:
            break

    if serialize_env:
        # Roundtrip serialization
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip.env.spec == env.env.spec

    if not spec.id in KNOWN_GYM_CLOSE_BROKEN:
        env.close()
    else:
        with test_case.assertRaisesRegex(
                AttributeError, "'MjViewer' object has no attribute 'finish'"):
            env.close()
