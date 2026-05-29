import gc
import pytest


@pytest.fixture()
def _gc_before_tmpdir_teardown(tmp_path):
    """Depend on tmp_path so this tears down before tmp_path cleanup."""
    yield
    for _ in range(3):
        gc.collect()


def pytest_collection_modifyitems(items):
    """Inject _gc_before_tmpdir_teardown into every test that uses tmp_path."""
    for item in items:
        if 'tmp_path' in item.fixturenames:
            item.fixturenames.append('_gc_before_tmpdir_teardown')
