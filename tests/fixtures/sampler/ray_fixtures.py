"""Pytest fixtures for intializing ray during ray related tests."""
import pytest
import ray


@pytest.fixture(scope='function')
def ray_local_session_fixture():
    """Initializes Ray and shuts down Ray in local mode.

    Yields:
        None: Yield is for purposes of pytest module style.
            All statements before the yield are apart of module setup, and all
            statements after the yield are apart of module teardown.

    """
    if not ray.is_initialized():
        ray.init(local_mode=True,
                 ignore_reinit_error=True,
                 log_to_driver=False)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope='function')
def ray_session_fixture():
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
                 log_to_driver=False)
    yield
    if ray.is_initialized():
        ray.shutdown()
