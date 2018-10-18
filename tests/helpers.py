import pickle

from tests.quirks import KNOWN_GYM_CLOSE_BROKEN
from tests.quirks import KNOWN_GYM_RENDER_NOT_IMPLEMENTED


def step_env(env, n=10, render=True):
    env.reset()
    for _ in range(n):
        a = env.action_space.sample()
        a_copy = a if isinstance(a, int) else a.copy()
        _, _, done, _ = env.step(a)

        # Check for side effects
        if isinstance(a, (int, float)):
            assert a == a_copy,\
            "Action was modified by environment!"
        else:
            assert a.all() == a_copy.all(),\
             "Action was modified by environment!"

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
        a = env.action_space.sample()
        a_copy = a if isinstance(a, int) else a.copy()
        _, _, done, _ = env.step(a)

        # Check for side effects
        if isinstance(a, (int, float)):
            assert a == a_copy,\
            "Action was modified by environment!"
        else:
            assert a.all() == a_copy.all(),\
            "Action was modified by environment!"

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
