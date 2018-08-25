import unittest


def _worker_collect_once(_):
    return 'a', 1


class TestStatefulPool(unittest.TestCase):
    def test_stateful_pool(self):
        from garage.sampler import stateful_pool
        stateful_pool.singleton_pool.initialize(n_parallel=3)
        results = stateful_pool.singleton_pool.run_collect(
            _worker_collect_once, 3, show_prog_bar=False)
        assert tuple(results) == ('a', 'a', 'a')

    def test_stateful_pool_over_capacity(self):
        from garage.sampler import stateful_pool
        stateful_pool.singleton_pool.initialize(n_parallel=4)
        results = stateful_pool.singleton_pool.run_collect(
            _worker_collect_once, 3, show_prog_bar=False)
        assert len(results) >= 3
