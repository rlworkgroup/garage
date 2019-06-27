import os
import os.path as osp

from nose2.tools import such

from garage.experiment.experiment import concretize, variant, VariantGenerator
from garage.experiment.experiment import ensure_dir, run_experiment

# https://gist.github.com/jrast/109f70f9b4c52bab4252


class TestClass:
    @property
    def arr(self):
        return [1, 2, 3]

    @property
    def compound_arr(self):
        return [dict(a=1)]


with such.A('instrument') as it:

    @it.should
    def test_concretize():
        it.assertEqual(concretize([5]), [5])
        it.assertEqual(concretize((5, )), (5, ))
        fake_globals = dict(TestClass=TestClass)
        modified = fake_globals['TestClass']
        it.assertEqual(concretize((5, )), (5, ))
        it.assertIsInstance(concretize(modified()), TestClass)

    @it.should
    def test_chained_call():
        fake_globals = dict(TestClass=TestClass)
        modified = fake_globals['TestClass']
        it.assertEqual(concretize(modified().arr[0]), 1)

    @it.should
    def test_variant_generator():

        vg = VariantGenerator()
        vg.add('key1', [1, 2, 3])
        vg.add('key2', [True, False])
        vg.add('key3', lambda key2: [1] if key2 else [1, 2])
        it.assertEqual(len(vg.variants()), 9)

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

        it.assertEqual(len(VG().variants()), 9)


it.createTests(globals())


def dummy_func(*_):
    pass


def test_default_log_dir():
    default_path = osp.join(os.getcwd(), 'data/local/experiment')
    ensure_dir(default_path)

    folder_num = len(os.listdir(default_path))

    run_experiment(dummy_func)

    assert len(os.listdir(default_path)) == folder_num + 1
