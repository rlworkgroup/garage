import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pytest

from garage.experiment.experiment import wrap_experiment


def _hard_rmtree(path):
    # Sometimes rmtree doesn't work, for some reason, but moving the directory
    # to a temporary directory does.
    shutil.rmtree(path, ignore_errors=True)
    try:
        with tempfile.TemporaryDirectory() as trash_dir:
            shutil.move(str(path), trash_dir)
    except FileNotFoundError:
        pass


def test_wrap_experiment_makes_log_dir():
    prefix = 'wrap_exp_test_makes_log_dir'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    _hard_rmtree(exp_path)
    expected_path = exp_path / 'test_exp'

    @wrap_experiment(prefix=prefix)
    def test_exp(ctxt=None):
        assert expected_path.samefile(ctxt.snapshot_dir)

    assert not exp_path.exists()

    test_exp()

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 1
    assert prefix_contents[0].samefile(expected_path)

    expected_path = exp_path / 'test_exp_1'

    test_exp()

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 2
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])

    expected_path = exp_path / 'test_exp_2'

    test_exp()

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 3
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])


def _run_launcher(launcher_path, prefix):
    with launcher_path.open('w') as launcher_f:
        launcher_f.write(
            textwrap.dedent(r"""
            from garage import wrap_experiment

            @wrap_experiment(prefix='{}')
            def test_exp(ctxt=None):
                print(ctxt.snapshot_dir)

            test_exp()""".format(prefix)))
    output = (subprocess.check_output(
        (sys.executable, str(launcher_path)),
        stderr=subprocess.STDOUT)).decode('utf-8').strip().split('\n')
    snapshot_dir = output[-1]
    return snapshot_dir, output


def test_wrap_experiment_builds_git_archive():
    prefix = 'wrap_exp_test_builds_git_archive'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    _hard_rmtree(exp_path)
    expected_path = exp_path / 'test_exp' / 'launch_archive.tar.xz'

    # Because __main__ actually points to pytest right now, we need to run the
    # "real" test in a subprocess.
    with tempfile.TemporaryDirectory() as launcher_dir:
        launch_dir = pathlib.Path(launcher_dir)
        subprocess.check_call(('git', 'init'), cwd=launcher_dir)
        # Make a test file, since git ls-files needs at least one commit.
        test_txt = launch_dir / 'test.txt'
        test_txt.touch()
        subprocess.check_call(('git', 'add', str(test_txt)), cwd=launcher_dir)
        subprocess.check_call(
            ('git', '-c', 'user.name=Test User', '-c',
             'user.email=test@example.com', 'commit', '-m', 'Initial commit'),
            cwd=launcher_dir)
        subdir = launch_dir / 'subdir'
        subdir.mkdir()
        launcher_path = pathlib.Path(launcher_dir) / 'subdir' / 'run_exp.py'

        snapshot_dir, _ = _run_launcher(launcher_path, prefix)

        archive_path = os.path.join(snapshot_dir, 'launch_archive.tar.xz')
        assert expected_path.samefile(archive_path)
        assert expected_path.exists()
        archive_size = expected_path.stat().st_size
        assert archive_size > 250
        contents = subprocess.check_output(
            ('tar', '--list', '--file', archive_path)).decode('utf-8')
        assert 'subdir/run_exp.py' in contents.strip()
        assert 'test.txt' in contents.strip()


def test_wrap_experiment_builds_git_archive_deleted_files():
    prefix = 'wrap_exp_test_builds_git_archive_deleted_files'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    _hard_rmtree(exp_path)
    expected_path = exp_path / 'test_exp' / 'launch_archive.tar.xz'

    # Because __main__ actually points to pytest right now, we need to run the
    # "real" test in a subprocess.
    with tempfile.TemporaryDirectory() as launcher_dir:
        launch_dir = pathlib.Path(launcher_dir)
        subprocess.check_call(('git', 'init'), cwd=launcher_dir)
        # Make a test file, since git ls-files needs at least one commit.
        to_delete = launch_dir / 'to_delete.txt'
        to_delete.touch()
        subprocess.check_call(('git', 'add', str(to_delete)), cwd=launcher_dir)
        subprocess.check_call(
            ('git', '-c', 'user.name=Test User', '-c',
             'user.email=test@example.com', 'commit', '-m', 'Initial commit'),
            cwd=launcher_dir)
        to_delete.unlink()
        subdir = launch_dir / 'subdir'
        subdir.mkdir()
        launcher_path = pathlib.Path(launcher_dir) / 'subdir' / 'run_exp.py'

        snapshot_dir, _ = _run_launcher(launcher_path, prefix)

        archive_path = os.path.join(snapshot_dir, 'launch_archive.tar.xz')
        assert expected_path.samefile(archive_path)
        assert expected_path.exists()
        archive_size = expected_path.stat().st_size
        assert archive_size > 250
        contents = subprocess.check_output(
            ('tar', '--list', '--file', archive_path)).decode('utf-8')
        assert 'subdir/run_exp.py' in contents.strip()
        assert 'test.txt' not in contents.strip()


def test_wrap_experiment_builds_git_archive_large_file():
    prefix = 'wrap_exp_test_builds_git_archive_large_files'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    _hard_rmtree(exp_path)
    expected_path = exp_path / 'test_exp' / 'launch_archive.tar.xz'

    # Because __main__ actually points to pytest right now, we need to run the
    # "real" test in a subprocess.
    with tempfile.TemporaryDirectory() as launcher_dir:
        launch_dir = pathlib.Path(launcher_dir)
        subprocess.check_call(('git', 'init'), cwd=launcher_dir)
        # Make a test file, since git ls-files needs at least one commit.
        test_txt = launch_dir / 'test.txt'
        test_txt.touch()
        subprocess.check_call(('git', 'add', str(test_txt)), cwd=launcher_dir)
        subprocess.check_call(
            ('git', '-c', 'user.name=Test User', '-c',
             'user.email=test@example.com', 'commit', '-m', 'Initial commit'),
            cwd=launcher_dir)
        subdir = launch_dir / 'subdir'
        subdir.mkdir()
        launcher_path = pathlib.Path(launcher_dir) / 'subdir' / 'run_exp.py'

        large_file = launch_dir / 'large.obj'
        with open(large_file, 'wb') as f:
            f.write(b'0' * int(1e7))

        snapshot_dir, output = _run_launcher(launcher_path, prefix)

        assert any(['archive_launch_repo' in line for line in output])

        archive_path = os.path.join(snapshot_dir, 'launch_archive.tar.xz')
        assert expected_path.samefile(archive_path)
        assert expected_path.exists()
        archive_size = expected_path.stat().st_size
        assert archive_size > 250
        contents = subprocess.check_output(
            ('tar', '--list', '--file', archive_path)).decode('utf-8')
        assert 'subdir/run_exp.py' in contents.strip()
        assert 'test.txt' in contents.strip()
        assert 'large.obj' not in contents.strip()


def test_wrap_experiment_launcher_outside_git():
    prefix = 'wrap_exp_test_launcher_outside_git'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    _hard_rmtree(exp_path)
    expected_path = exp_path / 'test_exp'

    # Because this is testing a file outside of a git repo, we need to make
    # ourselves a launcher script outside of any git repo.
    with tempfile.TemporaryDirectory() as launcher_dir:
        launcher_path = pathlib.Path(launcher_dir) / 'run_exp.py'
        snapshot_dir, _ = _run_launcher(launcher_path, prefix)
        assert os.path.samefile(str(expected_path), str(snapshot_dir))


def test_wrap_experiment_raises_on_non_ctxt_param_name():
    prefix = 'wrap_exp_test_prefix2'
    with pytest.raises(ValueError,
                       match="named 'ctxt' instead of '_snapshot_config'"):

        @wrap_experiment(prefix=prefix)
        def _test_exp(_snapshot_config=None):
            pass


def test_wrap_experiment_raises_on_empty_params():
    prefix = 'wrap_exp_test_prefix3'
    with pytest.raises(ValueError, match="named 'ctxt'"):

        @wrap_experiment(prefix=prefix)
        def _test_exp():
            pass


def test_wrap_experiment_name_parameters_passed():
    prefix = 'wrap_exp_test_name_parameters_passed'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    _hard_rmtree(exp_path)
    expected_path = exp_path / 'test_exp_seed=2'

    @wrap_experiment(prefix=prefix, name_parameters='passed')
    def test_exp(ctxt=None, seed=1):
        del seed
        assert expected_path.samefile(ctxt.snapshot_dir)

    assert not exp_path.exists()

    test_exp(seed=2)

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 1
    assert prefix_contents[0].samefile(expected_path)

    expected_path = exp_path / 'test_exp_seed=2_1'

    test_exp(seed=2)

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 2
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])

    expected_path = exp_path / 'test_exp_seed=3'

    test_exp(seed=3)

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 3
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])


def test_wrap_experiment_name_parameters_all():
    prefix = 'wrap_exp_test_name_parameters_all'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    _hard_rmtree(exp_path)
    expected_path = exp_path / 'test_exp_seed=1'

    @wrap_experiment(prefix=prefix, name_parameters='all')
    def test_exp(ctxt=None, seed=1):
        del seed
        assert expected_path.samefile(ctxt.snapshot_dir)

    assert not exp_path.exists()

    test_exp()

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 1
    assert prefix_contents[0].samefile(expected_path)

    expected_path = exp_path / 'test_exp_seed=1_1'

    test_exp()

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 2
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])

    expected_path = exp_path / 'test_exp_seed=1_2'

    test_exp()

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 3
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])


def test_wrap_experiment_name_parameters_all_disordered():
    prefix = 'wrap_exp_test_name_parameters_all_disordered'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    _hard_rmtree(exp_path)
    expected_path = exp_path / 'test_exp_seed=1_env=test-env'

    @wrap_experiment(prefix=prefix, name_parameters='all')
    def test_exp(ctxt=None, seed=1, env='test-env'):
        del seed
        del env
        assert expected_path.samefile(ctxt.snapshot_dir)

    assert not exp_path.exists()

    test_exp()

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 1
    assert prefix_contents[0].samefile(expected_path)

    expected_path = exp_path / 'test_exp_seed=2_env=test-env-v2'

    test_exp(env='test-env-v2', seed=2)

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 2
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])

    expected_path = exp_path / 'test_exp_seed=1_env=test-env-v2'

    test_exp(env='test-env-v2')

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 3
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])


def test_wrap_experiment_name_parameters_passed_disordered():
    prefix = 'wrap_exp_test_name_parameters_passed_disordered'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    _hard_rmtree(exp_path)
    expected_path = exp_path / 'test_exp_seed=2_env=test-env'

    @wrap_experiment(prefix=prefix, name_parameters='passed')
    def test_exp(ctxt=None, seed=1, env='test-env'):
        del seed
        del env
        assert expected_path.samefile(ctxt.snapshot_dir)

    assert not exp_path.exists()

    test_exp(seed=2, env='test-env')

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 1
    assert prefix_contents[0].samefile(expected_path)

    expected_path = exp_path / 'test_exp_seed=2_env=test-env-v2'

    test_exp(env='test-env-v2', seed=2)

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 2
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])

    expected_path = exp_path / 'test_exp_env=test-env-v2'

    test_exp(env='test-env-v2')

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 3
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])


def test_wrap_experiment_name_parameters_none():

    @wrap_experiment(name_parameters='none')
    def test_exp(ctxt=None, seed=1):
        del ctxt
        del seed

    with pytest.raises(ValueError, match='wrap_experiment.name_parameters'):
        test_exp()


def test_wrap_experiment_logdir():
    prefix = 'wrap_exp_logdir'
    name = 'specified_logdir'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    expected_path = exp_path / name
    _hard_rmtree(exp_path)
    logdir = 'data/local/wrap_exp_logdir/specified_logdir'

    @wrap_experiment(prefix=prefix, log_dir=logdir)
    def test_exp(ctxt=None, seed=1):
        del seed
        assert expected_path.samefile(ctxt.snapshot_dir)

    assert not exp_path.exists()

    test_exp()

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 1
    assert prefix_contents[0].samefile(expected_path)

    expected_path = pathlib.Path(os.getcwd(), logdir + '_1')

    test_exp()

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 2
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])


def test_wrap_experiment_dynamic_log_dir():
    prefix = 'wrap_exp_dynamic_logdir'
    name = 'specified_logdir'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    expected_path = exp_path / name
    _hard_rmtree(exp_path)
    logdir = 'data/local/wrap_exp_dynamic_logdir/specified_logdir'

    @wrap_experiment
    def test_exp(ctxt=None, seed=1):
        del seed
        assert expected_path.samefile(ctxt.snapshot_dir)

    assert not exp_path.exists()

    test_exp(dict(prefix=prefix, log_dir=logdir))

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 1
    assert prefix_contents[0].samefile(expected_path)

    expected_path = pathlib.Path(os.getcwd(), logdir + '_1')

    test_exp(dict(prefix=prefix, log_dir=logdir))

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 2
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])


def test_wrap_experiment_use_existing_dir():
    prefix = 'wrap_exp_existing_dir'
    name = 'test_exp'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    expected_path = exp_path / name
    _hard_rmtree(exp_path)
    logdir = 'data/local/wrap_exp_existing_dir/test_exp'

    @wrap_experiment(prefix=prefix)
    def test_exp(ctxt=None, seed=1):
        del seed
        assert expected_path.samefile(ctxt.snapshot_dir)

    assert not exp_path.exists()

    test_exp()

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 1
    assert prefix_contents[0].samefile(expected_path)

    @wrap_experiment(prefix=prefix)
    def test_exp_2(ctxt, seed=1):
        del seed
        assert expected_path.samefile(ctxt.snapshot_dir)

    test_exp_2(dict(log_dir=logdir, use_existing_dir=True))

    prefix_contents = list(exp_path.iterdir())
    assert len(prefix_contents) == 1
    assert any(
        [expected_path.samefile(directory) for directory in prefix_contents])


def test_wrap_experiment_invalid_options():
    prefix = 'wrap_exp_invalid_options'
    exp_path = pathlib.Path(os.getcwd(), 'data/local', prefix)
    _hard_rmtree(exp_path)
    logdir = 'data/local/wrap_exp_invalid_options/test_exp'

    @wrap_experiment(prefix=prefix)
    def test_exp(ctxt):
        del ctxt

    with pytest.raises(ValueError):
        test_exp(dict(logdir=logdir))
