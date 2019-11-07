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


def enumerate_algo_examples():
    """Return a list of paths for all algo examples

    Returns:
        List[str]: list of path strings

    """
    all_examples = EXAMPLES_ROOT_DIR.glob('**/*.py')
    return [str(e) for e in all_examples if e not in NON_ALGO_EXAMPLES]


@pytest.mark.no_cover
@pytest.mark.parametrize('filepath', enumerate_algo_examples())
def test_algo_examples(filepath):
    """Test algo examples.

    Args:
        filepath (str): path string of example

    """
    # Don't use check=True, since that causes subprocess to throw an error
    # in case of failure before the assertion is evaluated
    assert subprocess.run([filepath, '--n_epochs', '1'],
                          check=False).returncode == 0


@pytest.mark.no_cover
def test_step_env():
    """Test step_env.py."""
    assert subprocess.run(
        [EXAMPLES_ROOT_DIR / 'step_env.py', '--n_steps', '1'],
        check=False).returncode == 0


@pytest.mark.no_cover
def test_step_dm_control_env():
    """Test step_dm_control_env.py."""
    assert subprocess.run(
        [EXAMPLES_ROOT_DIR / 'step_dm_control_env.py', '--n_steps', '1'],
        check=False).returncode == 0
