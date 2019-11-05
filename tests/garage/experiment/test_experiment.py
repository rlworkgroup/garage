import os
import pathlib
import shutil

import pytest

from garage.experiment.experiment import run_experiment, wrap_experiment


def dummy_func(*_):
    pass


def test_default_log_dir():
    # Because this test uses the default log directory, if any other tests are
    # run in parallel with it that use this directory, it will fail.
    # For this reason, all tests which use run_experiment, must pass a
    # non-default exp_prefix or log_dir.
    default_path = os.path.join(os.getcwd(), 'data/local/experiment')
    pathlib.Path(default_path).mkdir(parents=True, exist_ok=True)

    folder_num = len(os.listdir(default_path))

    run_experiment(dummy_func)

    assert len(os.listdir(default_path)) == folder_num + 1


def test_experiment_with_none_task():
    with pytest.raises(Exception):
        run_experiment()


def test_experiment_with_not_callable_task():
    with pytest.raises(ValueError):
        run_experiment(1)


def test_experiment_with_variant():
    # Note: exp_name in variant does nothing.
    exp_variant = {'exp_name': 'test_name'}
    exp_path = os.path.join(os.getcwd(), 'data/local/test-prefix')
    pathlib.Path(exp_path).mkdir(parents=True, exist_ok=True)

    old_folder_contents = set(os.listdir(exp_path))
    # Pass a non-default exp_prefix, so test_default_log_dir is safe.
    run_experiment(dummy_func, exp_prefix='test_prefix', variant=exp_variant)
    new_folder_contents = set(os.listdir(exp_path))
    folder_content_diff = new_folder_contents - old_folder_contents
    assert len(folder_content_diff) == 1
    exp_folder_name = folder_content_diff.pop()
    assert exp_folder_name.startswith('test_prefix')


# Pylint gets confused by @wrap_experiment adding an argument.
# pylint: disable=no-value-for-parameter
def test_wrap_experiment_makes_log_dir():
    prefix = 'wrap_exp_test_prefix'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    shutil.rmtree(exp_path, ignore_errors=True)
    expected_path = exp_path / 'test_exp'

    @wrap_experiment(prefix=prefix)
    def test_exp(ctxt):
        assert expected_path.samefile(ctxt.snapshot_dir)

    with pytest.raises(FileNotFoundError):
        list(exp_path.iterdir())

    test_exp()

    new_folder_contents = list(exp_path.iterdir())
    assert len(new_folder_contents) == 1
    assert new_folder_contents[0].samefile(expected_path)

    expected_path = exp_path / 'test_exp_1'

    test_exp()

    new_folder_contents = list(exp_path.iterdir())
    assert len(new_folder_contents) == 2
    assert any([
        expected_path.samefile(directory) for directory in new_folder_contents
    ])

    expected_path = exp_path / 'test_exp_2'

    test_exp()

    new_folder_contents = list(exp_path.iterdir())
    assert len(new_folder_contents) == 3
    assert any([
        expected_path.samefile(directory) for directory in new_folder_contents
    ])
