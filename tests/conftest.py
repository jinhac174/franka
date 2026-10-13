"""Test setup: repo root on sys.path / as cwd, isolated LCM, headless rendering.

Run everything:            pytest
Fast unit tests only:      pytest -m "not integration"
One area:                  pytest tests/integration/test_sim_grasping.py -v

Integration tests run the processes in real time. On a heavily loaded shared
machine (see the load line in the pytest header) they can time out; a failure
there is worth rerunning on its own before suspecting the code. Process logs
of each test are kept under /tmp/pytest-of-$USER/pytest-current/<test>/.
"""

import os
import random
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

# Isolated LCM bus for this test session (processes started by the tests
# inherit it): tests never talk to a sim / robot you have running, nor to
# other users of the machine.
os.environ.setdefault(
    "LCM_DEFAULT_URL", f"udpm://239.255.76.67:{random.randint(20000, 29999)}?ttl=0"
)
# Headless offscreen rendering for camera envs if no display is available.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")


def pytest_report_header(config):
    """Integration tests run the sim / controllers in real time: on an
    overloaded machine they slow down and can time out. Say so up front."""
    try:
        load, n = os.getloadavg()[0], os.cpu_count() or 1
    except OSError:
        return None
    msg = f"load average {load:.0f} on {n} CPUs"
    if load > n:
        msg += ("  -- OVERLOADED: real-time integration tests may time out; "
                "rerun the failing ones alone or later")
    return msg


@pytest.fixture
def repo_root():
    return REPO
