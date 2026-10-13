import pytest

from tests.helpers import Listener, Procs, decoders


@pytest.fixture
def procs(tmp_path):
    with Procs(tmp_path) as p:
        yield p


@pytest.fixture
def lcm_listener():
    return Listener(decoders())


def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
