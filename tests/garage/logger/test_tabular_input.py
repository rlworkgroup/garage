import math
import unittest

from garage.logger import TabularInput
from garage.logger.tabular_input import TabularInputWarning


class TestTabularInput(unittest.TestCase):
    def setUp(self):
        self.tabular = TabularInput()

    def test_str(self):
        foo = 123
        bar = 456
        baz = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.tabular.record('foo', foo)
        self.tabular.record('bar', bar)
        self.tabular.record('baz', baz)

        correct_str = (
            '---  ---\n'
            'bar  456\n'
            'foo  123\n'
            '---  ---'
        )  # yapf: disable
        assert str(self.tabular) == correct_str

    def test_record(self):
        foo = 1
        bar = 10
        self.tabular.record('foo', foo)
        self.tabular.record('bar', bar)

        assert self.tabular.as_dict['foo'] == foo
        assert self.tabular.as_dict['bar'] == bar

    def test_record_misc_stat(self):
        self.tabular.record_misc_stat('Foo', [0, 1, 2])
        bar = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.tabular.record_misc_stat('Bar', bar, placement='front')

        correct = {
            'FooAverage': 1.0,
            'FooStd': 0.816496580927726,
            'FooMedian': 1.0,
            'FooMin': 0,
            'FooMax': 2,
            'AverageBar': 5.5,
            'StdBar': 2.8722813232690143,
            'MedianBar': 5.5,
            'MinBar': 1,
            'MaxBar': 10,
        }
        self.assertDictEqual(self.tabular.as_dict, correct)

    def test_record_misc_stat_nan(self):
        self.tabular.record_misc_stat('none', None)

        correct = {
            'noneAverage': math.nan,
            'noneStd': math.nan,
            'noneMedian': math.nan,
            'noneMin': math.nan,
            'noneMax': math.nan
        }
        for k, v in self.tabular.as_dict.items():
            assert correct[k] is math.nan

    def test_prefix(self):
        foo = 111
        bar = 222
        with self.tabular.prefix('test_'):
            self.tabular.record('foo', foo)
            self.tabular.record('bar', bar)

        correct = {'test_foo': foo, 'test_bar': bar}
        self.assertDictEqual(self.tabular.as_dict, correct)

    def test_clear(self):
        foo = 1
        bar = 10
        self.tabular.record('foo', foo)
        self.tabular.record('bar', bar)
        self.tabular.mark_all()

        assert self.tabular.as_dict
        self.tabular.clear()
        assert not self.tabular.as_dict

    def test_clear_warns_not_recorded_once(self):
        self.tabular.record('foo', 1)

        with self.assertWarns(TabularInputWarning):
            self.tabular.clear()

        self.tabular.record('foo', 1)
        # This not trigger a warning, because we warned once
        self.tabular.clear()

    def test_disable_warnings(self):
        self.tabular.record('foo', 1)

        with self.assertWarns(TabularInputWarning):
            self.tabular.clear()

        self.tabular.record('bar', 2)
        self.tabular.disable_warnings()

        # This should not trigger a warning, because we disabled warnings
        self.tabular.clear()

    def test_push_prefix(self):
        foo = 111
        bar = 222
        self.tabular.push_prefix('aaa_')
        self.tabular.record('foo', foo)
        self.tabular.record('bar', bar)
        self.tabular.push_prefix('bbb_')
        self.tabular.record('foo', foo)
        self.tabular.record('bar', bar)

        correct = {
            'aaa_foo': foo,
            'aaa_bar': bar,
            'aaa_bbb_foo': foo,
            'aaa_bbb_bar': bar,
        }
        self.assertDictEqual(self.tabular.as_dict, correct)

    def test_pop_prefix(self):
        foo = 111
        bar = 222

        self.tabular.push_prefix('aaa_')
        self.tabular.record('foo', foo)
        self.tabular.record('bar', bar)
        self.tabular.push_prefix('bbb_')
        self.tabular.record('foo', foo)
        self.tabular.record('bar', bar)
        self.tabular.pop_prefix()
        self.tabular.record('foopop', foo)
        self.tabular.record('barpop', bar)
        self.tabular.pop_prefix()
        self.tabular.record('foopop', foo)
        self.tabular.record('barpop', bar)
        correct = {
            'aaa_foo': foo,
            'aaa_bar': bar,
            'aaa_bbb_foo': foo,
            'aaa_bbb_bar': bar,
            'aaa_foopop': foo,
            'aaa_barpop': bar,
            'foopop': foo,
            'barpop': bar,
        }
        self.assertDictEqual(self.tabular.as_dict, correct)

    def test_as_primitive_dict(self):
        stuff = {
            'int': int(1),
            'float': float(2.0),
            'bool': bool(True),
            'str': str('Hello, world!'),
            'dict': dict(foo='bar'),
        }
        for k, v in stuff.items():
            self.tabular.record(k, v)

        correct = {
            'int': int(1),
            'float': float(2.0),
            'bool': bool(True),
            'str': str('Hello, world!'),
        }
        self.assertDictEqual(self.tabular.as_primitive_dict, correct)
