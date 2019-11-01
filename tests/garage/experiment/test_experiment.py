import os
import pathlib

import pytest

from garage.experiment.experiment import run_experiment
from garage.experiment.experiment import variant, VariantGenerator


class TestExperiment:

    def test_variant_generator(self):

        vg = VariantGenerator()
        vg.add('key1', [1, 2, 3])
        vg.add('key2', [True, False])
        vg.add('key3', lambda key2: [1] if key2 else [1, 2])
        assert len(vg.variants()) == 9

        class VG(VariantGenerator):

            @variant
            def key1(self):
                return [1, 2, 3]

            @variant
            def key2(self):
                yield True
                yield False

            @variant
            def key3(self, key2):
                if key2:
                    yield 1
                else:
                    yield 1
                    yield 2

        assert len(VG().variants()) == 9


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
