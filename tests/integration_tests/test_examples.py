"""This is an integration test to make sure scripts from examples/
work when executing `./examples/**/*.py`.
"""
import os
import pathlib
import subprocess

import pytest

EXAMPLES_ROOT_DIR = pathlib.Path('examples/')
NON_ALGO_EXAMPLES = [
    EXAMPLES_ROOT_DIR / 'resume_training.py',
    EXAMPLES_ROOT_DIR / 'sim_policy.py',
    EXAMPLES_ROOT_DIR / 'step_env.py',
    EXAMPLES_ROOT_DIR / 'step_dm_control_env.py',
]

LONG_RUNNING_EXAMPLES = [
    EXAMPLES_ROOT_DIR / 'tf/ppo_memorize_digits.py',
    EXAMPLES_ROOT_DIR / 'tf/dqn_pong.py',
    EXAMPLES_ROOT_DIR / 'tf/trpo_cubecrash.py',
]


def enumerate_algo_examples():
    """Return a list of paths for all algo examples

    Returns:
        List[str]: list of path strings

    """
    exclude = NON_ALGO_EXAMPLES + LONG_RUNNING_EXAMPLES
    all_examples = EXAMPLES_ROOT_DIR.glob('**/*.py')
    return [str(e) for e in all_examples if e not in exclude]


@pytest.mark.no_cover
@pytest.mark.timeout(70)
@pytest.mark.parametrize('filepath', enumerate_algo_examples())
def test_algo_examples(filepath):
    """Test algo examples.

    Args:
        filepath (str): path string of example

    """
    if filepath == str(EXAMPLES_ROOT_DIR / 'tf/her_ddpg_fetchreach.py'):
        pytest.skip('Temporarily skipped because it is broken')

    env = os.environ.copy()
    env['GARAGE_EXAMPLE_TEST_N_EPOCHS'] = '1'
    # Don't use check=True, since that causes subprocess to throw an error
    # in case of failure before the assertion is evaluated
    assert subprocess.run([filepath], check=False, env=env).returncode == 0


@pytest.mark.no_cover
@pytest.mark.timeout(180)
def test_dqn_pong():
    """Test tf/dqn_pong.py with reduced replay buffer size for reduced memory
    consumption.
    """
    env = os.environ.copy()
    env['GARAGE_EXAMPLE_TEST_N_EPOCHS'] = '1'
    assert subprocess.run(
        [str(EXAMPLES_ROOT_DIR / 'tf/dqn_pong.py'), '--buffer_size', '5'],
        check=False,
        env=env).returncode == 0


@pytest.mark.no_cover
@pytest.mark.timeout(30)
def test_ppo_memorize_digits():
    """Test tf/ppo_memorize_digits.py with reduced batch size for reduced
    memory consumption.
    """
    env = os.environ.copy()
    env['GARAGE_EXAMPLE_TEST_N_EPOCHS'] = '1'
    command = [
        str(EXAMPLES_ROOT_DIR / 'tf/ppo_memorize_digits.py'), '--batch_size',
        '4'
    ]
    assert subprocess.run(command, check=False, env=env).returncode == 0


@pytest.mark.no_cover
@pytest.mark.timeout(40)
def test_trpo_cubecrash():
    """Test tf/trpo_cubecrash.py with reduced batch size for reduced memory
    consumption.
    """
    env = os.environ.copy()
    env['GARAGE_EXAMPLE_TEST_N_EPOCHS'] = '1'
    assert subprocess.run(
        [str(EXAMPLES_ROOT_DIR / 'tf/trpo_cubecrash.py'), '--batch_size', '4'],
        check=False,
        env=env).returncode == 0


@pytest.mark.no_cover
@pytest.mark.timeout(10)
def test_step_env():
    """Test step_env.py."""
    assert subprocess.run(
        [EXAMPLES_ROOT_DIR / 'step_env.py', '--n_steps', '1'],
        check=False).returncode == 0


@pytest.mark.no_cover
@pytest.mark.timeout(20)
def test_step_dm_control_env():
    """Test step_dm_control_env.py."""
    assert subprocess.run(
        [EXAMPLES_ROOT_DIR / 'step_dm_control_env.py', '--n_steps', '1'],
        check=False).returncode == 0
