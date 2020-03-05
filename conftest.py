"""Place Pytest fixtures to be used across all tests here."""
import pytest
import ray


@pytest.fixture(scope='module')
def ray_test_fixture():
    """Initializes Ray and shuts down Ray.

    Yields:
        None: Yield is for purposes of pytest module style.
            All statements before the yield are apart of module setup, and all
            statements after the yield are apart of module teardown.
    """
    if not ray.is_initialized():
        ray.init(memory=52428800,
                 object_store_memory=78643200,
                 ignore_reinit_error=True,
                 log_to_driver=False,
                 include_webui=False)
    yield
    if ray.is_initialized():
        ray.shutdown()
