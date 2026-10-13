"""Tutorials 06, 07 and 09 end-to-end: Cartesian IK-QP teleop, min-jerk
point-to-point with RRT, and IK reach + joint impedance (headless sim)."""

import re
import time

import numpy as np

from tests.helpers import STARTUP, CTRL, ENV, TcpFK, arm_q, wait_settled

E = "configs/franka_fr3/env/collision.yaml"
C = "configs/franka_fr3/controllers"


def _start(procs, lst, ctrl_cfg, *extra):
    procs.start("env", ENV, E)
    assert lst.wait_for(lambda: "franka_arm_joint_meas" in lst.last, STARTUP), "env did not publish"
    ctrl = procs.start("ctrl", CTRL, f"{C}/{ctrl_cfg}", *extra)
    assert lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, STARTUP), "no controller"
    lst.spin(1.0)
    return ctrl


def _wait_log(proc, pattern, timeout, since=0):
    """First match of ``pattern`` in the process log after offset ``since``."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = re.search(pattern, proc.log()[since:])
        if m:
            return m
        time.sleep(0.2)
    raise AssertionError(f"{pattern!r} not in log:\n{proc.log()[-3000:]}")


def test_ik_qp_cartesian_nudges(procs, lcm_listener):
    """w x3 (+3 cm x), r x2 (+2 cm z): the TCP follows."""
    lst, fk = lcm_listener, TcpFK()
    ctrl = _start(procs, lst, "ik_qp.yaml")
    p0 = fk(arm_q(lst))
    for k in "wwwrr":
        ctrl.key(k)
        time.sleep(0.3)
    assert wait_settled(lst, 15)
    dp = fk(arm_q(lst)) - p0
    assert np.allclose(dp, [0.03, 0.0, 0.02], atol=0.006), dp


def test_min_jerk_sample_plan_track(procs, lcm_listener):
    lst = lcm_listener
    ctrl = _start(procs, lst, "min_jerk_p2p.yaml")
    ctrl.key("1")
    m = _wait_log(ctrl, r"Sampled collision-free goal \(arm=\[([^\]]+)\]", 30)
    q_goal = np.array([float(x) for x in m.group(1).split(",")])
    ctrl.key("2")
    _wait_log(ctrl, r"Plan OK", 60)
    ctrl.key("3")
    _wait_log(ctrl, r"Tracking armed", 10)
    time.sleep(1.0)
    assert wait_settled(lst, 30)
    assert np.max(np.abs(arm_q(lst) - q_goal)) < 0.01  # printed with 3 decimals


def test_ik_reach_sample_solve_go(procs, lcm_listener):
    """1 (sample a TCP pose) -> 2 (IK; resample if it fails / collides) ->
    3 (joint impedance to q*): the TCP ends at the sampled pose."""
    lst, fk = lcm_listener, TcpFK()
    ctrl = _start(procs, lst, "ik_reach.yaml", "--sample_seed 3")
    for _ in range(6):
        n = len(ctrl.log())
        ctrl.key("1")
        m = _wait_log(ctrl, r"Sampled fr3_gripper_tcp p=\[([^\]]+)\]", 30, n)
        p_goal = np.array([float(x) for x in m.group(1).split()])
        n = len(ctrl.log())
        ctrl.key("2")
        r = _wait_log(ctrl, r"IK OK \(.*\), collision-free|IK failed|in collision", 30, n)
        if "collision-free" in r.group(0):
            break
    else:
        raise AssertionError("no reachable sample in 6 tries")
    ctrl.key("3")
    _wait_log(ctrl, r"Joint impedance", 10)
    time.sleep(1.0)
    assert wait_settled(lst, 30)
    assert np.linalg.norm(fk(arm_q(lst)) - p_goal) < 0.005
