from garage.sampler.stateful_pool import StatefulPool


def _worker_collect_once(_):
    return 'a', 1


class TestStatefulPool:
    def test_stateful_pool(self):
        stateful_pool = StatefulPool()
        stateful_pool.initialize(n_parallel=10)
        results = stateful_pool.run_collect(
            _worker_collect_once, 3, show_prog_bar=False)
        assert all([r == 'a' for r in results]) and len(results) >= 3

    def test_stateful_pool_over_capacity(self):
        stateful_pool = StatefulPool()
        stateful_pool.initialize(n_parallel=4)
        results = stateful_pool.run_collect(
            _worker_collect_once, 3, show_prog_bar=False)
        assert len(results) >= 3
