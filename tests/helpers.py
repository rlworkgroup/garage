import pickle
import unittest

import glfw
from mujoco_py.mjviewer import MjViewer
import tensorflow as tf

from tests.quirks import KNOWN_GYM_CLOSE_BROKEN
from tests.quirks import KNOWN_GYM_NOT_CLOSE_VIEWER
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


def step_env_with_gym_quirks(test_case, env, spec, n=10, render=True):
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
    if not spec.id in KNOWN_GYM_CLOSE_BROKEN:
        env.close()
    else:
        with test_case.assertRaisesRegex(
                AttributeError, "'MjViewer' object has no attribute 'finish'"):
            env.close()
    if any([s in spec._entry_point for s in KNOWN_GYM_NOT_CLOSE_VIEWER]):
        close_viewer_window(env)


def pickle_env_with_gym_quirks(test_case, env, spec, render=True):
    # Roundtrip serialization
    round_trip = pickle.loads(pickle.dumps(env))
    assert round_trip.env.spec == env.env.spec
    env = round_trip

    # Step and render still work after pickling
    env.reset()
    env.step(env.action_space.sample())

    if render:
        if not spec.id in KNOWN_GYM_RENDER_NOT_IMPLEMENTED:
            env.render()
        else:
            with test_case.assertRaises(NotImplementedError):
                env.render()

    round_trip = pickle.loads(pickle.dumps(env))
    assert round_trip.env.spec == env.env.spec

    if spec.id not in KNOWN_GYM_CLOSE_BROKEN:
        env.close()
    else:
        with test_case.assertRaisesRegex(
                AttributeError, "'MjViewer' object has no attribute 'finish'"):
            env.close()
    if any([s in spec._entry_point for s in KNOWN_GYM_NOT_CLOSE_VIEWER]):
        close_viewer_window(env)


def close_viewer_window(env):
    if hasattr(env, "viewer") and isinstance(env.viewer, MjViewer):
        glfw.destroy_window(env.viewer.window)
    else:
        env_itr = env
        while hasattr(env_itr, "env"):
            env_itr = env_itr.env
            if (hasattr(env_itr, "viewer")
                    and isinstance(env_itr.viewer, MjViewer)):
                glfw.destroy_window(env_itr.viewer.window)
                break
