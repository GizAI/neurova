import gc
import pytest


def pytest_runtest_teardown(item, nextitem):
    """Force garbage collection after each test to close SQLite connections."""
    gc.collect()
