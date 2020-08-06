"""Integration tests to make sure scripts in examples work."""
import os
import pathlib
import subprocess

import pytest

EXAMPLES_ROOT_DIR = pathlib.Path('examples/')
NON_ALGO_EXAMPLES = [
    EXAMPLES_ROOT_DIR / 'torch/resume_training.py',
    EXAMPLES_ROOT_DIR / 'tf/resume_training.py',
    EXAMPLES_ROOT_DIR / 'sim_policy.py',
    EXAMPLES_ROOT_DIR / 'step_env.py',
    EXAMPLES_ROOT_DIR / 'step_dm_control_env.py',
]

# yapf: disable
LONG_RUNNING_EXAMPLES = [
    EXAMPLES_ROOT_DIR / 'tf/ppo_memorize_digits.py',
    EXAMPLES_ROOT_DIR / 'tf/dqn_pong.py',
    EXAMPLES_ROOT_DIR / 'tf/her_ddpg_fetchreach.py',
    EXAMPLES_ROOT_DIR / 'tf/trpo_cubecrash.py',
    EXAMPLES_ROOT_DIR / 'torch/maml_ppo_half_cheetah_dir.py',
    EXAMPLES_ROOT_DIR / 'torch/maml_trpo_half_cheetah_dir.py',
    EXAMPLES_ROOT_DIR / 'torch/maml_vpg_half_cheetah_dir.py',
    EXAMPLES_ROOT_DIR / 'torch/maml_trpo_metaworld_ml1_push.py',
    EXAMPLES_ROOT_DIR / 'torch/maml_trpo_metaworld_ml10.py',
    EXAMPLES_ROOT_DIR / 'torch/maml_trpo_metaworld_ml45.py',
    EXAMPLES_ROOT_DIR / 'torch/pearl_half_cheetah_vel.py',
    EXAMPLES_ROOT_DIR / 'torch/pearl_metaworld_ml1_push.py',
    EXAMPLES_ROOT_DIR / 'torch/pearl_metaworld_ml10.py',
    EXAMPLES_ROOT_DIR / 'torch/pearl_metaworld_ml45.py',
    EXAMPLES_ROOT_DIR / 'tf/rl2_ppo_metaworld_ml1_push.py',
    EXAMPLES_ROOT_DIR / 'tf/rl2_ppo_metaworld_ml10.py',
    EXAMPLES_ROOT_DIR / 'tf/rl2_ppo_metaworld_ml10_meta_test.py',
    EXAMPLES_ROOT_DIR / 'tf/rl2_ppo_metaworld_ml45.py',
    EXAMPLES_ROOT_DIR / 'torch/mtsac_metaworld_mt10.py',
    EXAMPLES_ROOT_DIR / 'torch/mtsac_metaworld_mt50.py',
    EXAMPLES_ROOT_DIR / 'torch/mtsac_metaworld_ml1_pick_place.py',
    EXAMPLES_ROOT_DIR / 'torch/mtppo_metaworld_ml1_push.py',
    EXAMPLES_ROOT_DIR / 'torch/mtppo_metaworld_mt10.py',
    EXAMPLES_ROOT_DIR / 'torch/mtppo_metaworld_mt50.py',
    EXAMPLES_ROOT_DIR / 'torch/mttrpo_metaworld_ml1_push.py',
    EXAMPLES_ROOT_DIR / 'torch/mttrpo_metaworld_mt10.py',
    EXAMPLES_ROOT_DIR / 'torch/mttrpo_metaworld_mt50.py',
    EXAMPLES_ROOT_DIR / 'tf/te_ppo_point.py',
    EXAMPLES_ROOT_DIR / 'tf/te_ppo_metaworld_ml1_push.py',
    EXAMPLES_ROOT_DIR / 'tf/te_ppo_metaworld_mt10.py',
    EXAMPLES_ROOT_DIR / 'tf/te_ppo_metaworld_mt50.py',
]
# yapf: enable


def enumerate_algo_examples():
    """Return a list of paths for all algo examples.

    Returns:
        List[str]: list of path strings

    """
    exclude = NON_ALGO_EXAMPLES + LONG_RUNNING_EXAMPLES
    all_examples = EXAMPLES_ROOT_DIR.glob('**/*.py')
    return [str(e) for e in all_examples if e not in exclude]


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(150)
@pytest.mark.parametrize('filepath', enumerate_algo_examples())
def test_algo_examples(filepath):
    """Test algo examples.

    Args:
        filepath (str): path string of example

    """
    env = os.environ.copy()
    env['GARAGE_EXAMPLE_TEST_N_EPOCHS'] = '1'
    # Don't use check=True, since that causes subprocess to throw an error
    # in case of failure before the assertion is evaluated
    assert subprocess.run([filepath], check=False, env=env).returncode == 0


@pytest.mark.no_cover
@pytest.mark.timeout(180)
def test_dqn_pong():
    """Test tf/dqn_pong.py with reduced replay buffer size.

    This is to reduced memory consumption.

    """
    env = os.environ.copy()
    env['GARAGE_EXAMPLE_TEST_N_EPOCHS'] = '1'
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'tf/dqn_pong.py', '--buffer_size', '5',
        '--max_episode_length', '5'
    ],
                          check=False,
                          env=env).returncode == 0


@pytest.mark.no_cover
@pytest.mark.timeout(30)
def test_ppo_memorize_digits():
    """Test tf/ppo_memorize_digits.py with reduced batch size.

    This is to reduced memory consumption.

    """
    env = os.environ.copy()
    env['GARAGE_EXAMPLE_TEST_N_EPOCHS'] = '1'
    command = [
        EXAMPLES_ROOT_DIR / 'tf/ppo_memorize_digits.py', '--batch_size', '4'
    ]
    assert subprocess.run(command, check=False, env=env).returncode == 0


@pytest.mark.no_cover
@pytest.mark.timeout(40)
def test_trpo_cubecrash():
    """Test tf/trpo_cubecrash.py with reduced batch size.

    This is to reduced memory consumption.

    """
    env = os.environ.copy()
    env['GARAGE_EXAMPLE_TEST_N_EPOCHS'] = '1'
    assert subprocess.run(
        [EXAMPLES_ROOT_DIR / 'tf/trpo_cubecrash.py', '--batch_size', '4'],
        check=False,
        env=env).returncode == 0


@pytest.mark.no_cover
@pytest.mark.timeout(10)
def test_step_gym_env():
    """Test step_env.py."""
    assert subprocess.run(
        [EXAMPLES_ROOT_DIR / 'step_gym_env.py', '--n_steps', '1'],
        check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(20)
def test_step_dm_control_env():
    """Test step_dm_control_env.py."""
    assert subprocess.run(
        [EXAMPLES_ROOT_DIR / 'step_dm_control_env.py', '--n_steps', '1'],
        check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(30)
def test_maml_halfcheetah():
    """Test maml_trpo_half_cheetah_dir.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/maml_trpo_half_cheetah_dir.py', '--epochs',
        '1', '--episodes_per_task', '1', '--meta_batch_size', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(60)
def test_pearl_half_cheetah_vel():
    """Test pearl_half_cheetah_vel.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/pearl_half_cheetah_vel.py', '--num_epochs',
        '1', '--num_train_tasks', '5', '--num_test_tasks', '1',
        '--encoder_hidden_size', '2', '--net_size', '2',
        '--num_steps_per_epoch', '5', '--num_initial_steps', '5',
        '--num_steps_prior', '1', '--num_extra_rl_steps_posterior', '1',
        '--batch_size', '4', '--embedding_batch_size', '2',
        '--embedding_mini_batch_size', '2', '--max_episode_length', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(60)
def test_pearl_metaworld_ml1_push():
    """Test pearl_ml1_push.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/pearl_metaworld_ml1_push.py',
        '--num_epochs', '1', '--num_train_tasks', '5', '--num_test_tasks', '1',
        '--encoder_hidden_size', '2', '--net_size', '2',
        '--num_steps_per_epoch', '5', '--num_initial_steps', '5',
        '--num_steps_prior', '1', '--num_extra_rl_steps_posterior', '1',
        '--batch_size', '4', '--embedding_batch_size', '2',
        '--embedding_mini_batch_size', '2', '--max_episode_length', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
def test_pearl_metaworld_ml10():
    """Test pearl_ml10.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/pearl_metaworld_ml10.py', '--num_epochs',
        '1', '--num_train_tasks', '1', '--num_test_tasks', '1',
        '--encoder_hidden_size', '1', '--net_size', '2',
        '--num_steps_per_epoch', '2', '--num_initial_steps', '2',
        '--num_steps_prior', '1', '--num_extra_rl_steps_posterior', '1',
        '--batch_size', '2', '--embedding_batch_size', '1',
        '--embedding_mini_batch_size', '1', '--max_episode_length', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.skip('Temporarily skipped because of out-of-memory error')
@pytest.mark.mujoco
@pytest.mark.no_cover
def test_pearl_metaworld_ml45():
    """Test pearl_ml45.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/pearl_metaworld_ml45.py', '--num_epochs',
        '1', '--num_train_tasks', '1', '--num_test_tasks', '1',
        '--encoder_hidden_size', '1', '--net_size', '2',
        '--num_steps_per_epoch', '2', '--num_initial_steps', '2',
        '--num_steps_prior', '1', '--num_extra_rl_steps_posterior', '1',
        '--batch_size', '2', '--embedding_batch_size', '1',
        '--embedding_mini_batch_size', '1', '--max_episode_length', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.nightly
@pytest.mark.no_cover
@pytest.mark.timeout(200)
def test_maml_trpo_metaworld_ml10():
    """Test maml_trpo_ml10.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/maml_trpo_metaworld_ml10.py', '--epochs',
        '1', '--episodes_per_task', '1', '--meta_batch_size', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(30)
def test_maml_trpo():
    """Test maml_trpo_half_cheetah_dir.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/maml_trpo_half_cheetah_dir.py', '--epochs',
        '1', '--episodes_per_task', '1', '--meta_batch_size', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(30)
def test_maml_ppo():
    """Test maml_ppo_half_cheetah_dir.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/maml_ppo_half_cheetah_dir.py', '--epochs',
        '1', '--episodes_per_task', '1', '--meta_batch_size', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(30)
def test_maml_vpg():
    """Test maml_vpg_half_cheetah_dir.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/maml_vpg_half_cheetah_dir.py', '--epochs',
        '1', '--episodes_per_task', '1', '--meta_batch_size', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.nightly
@pytest.mark.no_cover
@pytest.mark.timeout(120)
def test_rl2_metaworld_ml1_push():
    """Test rl2_ppo_ml1.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'tf/rl2_ppo_metaworld_ml1_push.py', '--n_epochs',
        '1', '--episode_per_task', '1', '--meta_batch_size', '10'
    ],
                          check=False).returncode == 0


@pytest.mark.nightly
@pytest.mark.no_cover
@pytest.mark.timeout(200)
def test_rl2_ppo_metaworld_ml10():
    """Test rl2_ppo_ml10.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'tf/rl2_ppo_metaworld_ml10.py', '--n_epochs', '1',
        '--episode_per_task', '1', '--meta_batch_size', '10'
    ],
                          check=False).returncode == 0


@pytest.mark.nightly
@pytest.mark.no_cover
@pytest.mark.timeout(200)
def test_rl2_ppo_metaworld_ml10_meta_test():
    """Test rl2_ppo_ml10_meta_test.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'tf/rl2_ppo_metaworld_ml10_meta_test.py',
        '--n_epochs', '1', '--episode_per_task', '1', '--meta_batch_size', '10'
    ],
                          check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(60)
def test_mtppo_metaworld_ml1_push():
    """Test ppo_metaworld_ml1_push.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/mtppo_metaworld_ml1_push.py', '--epochs',
        '1', '--batch_size', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(60)
def test_mtppo_metaworld_mt10():
    """Test ppo_metaworld_mt10.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/mtppo_metaworld_mt10.py', '--epochs', '1',
        '--batch_size', '1', '--n_worker', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.skip('Temporarily skipped because it takes too long time')
@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(300)
def test_mtppo_metaworld_mt50():
    """Test ppo_metaworld_mt50.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/mtppo_metaworld_mt50.py', '--epochs', '1',
        '--batch_size', '1', '--n_worker', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(60)
def test_trpo_metaworld_ml1_push():
    """Test trpo_metaworld_ml1_push.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/mttrpo_metaworld_ml1_push.py', '--epochs',
        '1', '--batch_size', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(60)
def test_trpo_metaworld_mt10():
    """Test trpo_metaworld_mt10.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/mttrpo_metaworld_mt10.py', '--epochs', '1',
        '--batch_size', '1', '--n_worker', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.skip('Temporarily skipped because it takes too long time')
@pytest.mark.mujoco
@pytest.mark.no_cover
@pytest.mark.timeout(300)
def test_trpo_metaworld_mt50():
    """Test trpo_metaworld_mt50.py."""
    assert subprocess.run([
        EXAMPLES_ROOT_DIR / 'torch/mttrpo_metaworld_mt50.py', '--epochs', '1',
        '--batch_size', '1', '--n_worker', '1'
    ],
                          check=False).returncode == 0


@pytest.mark.no_cover
@pytest.mark.timeout(60)
def test_te_ppo_point():
    """Test te_ppo_point.py."""
    # yapf: disable
    assert subprocess.run([
        str(EXAMPLES_ROOT_DIR / 'tf/te_ppo_point.py'), '--n_epochs', '1',
        '--batch_size_per_task', '100'
    ], check=False).returncode == 0
    # yapf: enable


@pytest.mark.no_cover
@pytest.mark.mujoco
@pytest.mark.timeout(100)
def test_te_ppo_metaworld_ml1_push():
    """Test te_ppo_point.py."""
    # yapf: disable
    assert subprocess.run([
        str(EXAMPLES_ROOT_DIR / 'tf/te_ppo_metaworld_ml1_push.py'),
        '--n_epochs', '1', '--batch_size_per_task', '100'
    ], check=False).returncode == 0
    # yapf: enable


@pytest.mark.no_cover
@pytest.mark.mujoco
@pytest.mark.timeout(300)
def test_te_ppo_metaworld_mt10():
    """Test te_ppo_point.py."""
    # yapf: disable
    assert subprocess.run([
        str(EXAMPLES_ROOT_DIR / 'tf/te_ppo_metaworld_mt10.py'),
        '--n_epochs', '1', '--batch_size_per_task', '100'
    ], check=False).returncode == 0
    # yapf: enable


@pytest.mark.skip('Temporarily skipped because it takes too long time')
@pytest.mark.no_cover
@pytest.mark.mujoco
@pytest.mark.timeout(300)
def test_te_ppo_metaworld_mt50():
    """Test te_ppo_point.py."""
    # yapf: disable
    assert subprocess.run([
        str(EXAMPLES_ROOT_DIR / 'tf/te_ppo_metaworld_mt50.py'),
        '--n_epochs', '1', '--batch_size_per_task', '100'
    ], check=False).returncode == 0
    # yapf: enable
