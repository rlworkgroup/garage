import os
import pathlib

import pytest

from garage.experiment.experiment import concretize, variant, VariantGenerator
from garage.experiment.experiment import run_experiment

# https://gist.github.com/jrast/109f70f9b4c52bab4252


class TestClass:
    @property
    def arr(self):
        return [1, 2, 3]

    @property
    def compound_arr(self):
        return [dict(a=1)]


class TestExperiment:
    def test_concretize(self):
        assert concretize([5]) == [5]
        assert concretize((5, )) == (5, )
        fake_globals = dict(TestClass=TestClass)
        modified = fake_globals['TestClass']
        assert concretize((5, )) == (5, )
        assert isinstance(concretize(modified()), TestClass)

    def test_chained_call(self):
        fake_globals = dict(TestClass=TestClass)
        modified = fake_globals['TestClass']
        assert concretize(modified().arr[0]) == 1

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
    variant = {'exp_name': 'test'}
    run_experiment(dummy_func, variant=variant)
