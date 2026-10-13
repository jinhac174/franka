"""Tutorials 01-05 and 08 end-to-end: MuJoCo env + controller processes over
LCM (headless). Each test starts the same commands the tutorial gives."""

import time

import lcm
import numpy as np

from data_type.basic_types.GripperCmdData import GripperCmdData
from lcm_type.gripper.gripper_cmd_t import gripper_cmd_t
from tests.helpers import STARTUP, CTRL, ENV, HOME_Q, arm_q

E = "configs/franka_fr3/env"
C = "configs/franka_fr3/controllers"


def _has_arm(lst):
    return lambda: "franka_arm_joint_meas" in lst.last


def test_grav_comp_holds_and_gripper_moves(procs, lcm_listener):
    lst = lcm_listener
    procs.start("env", ENV, f"{E}/default.yaml")
    assert lst.wait_for(_has_arm(lst), STARTUP), "env did not publish"
    procs.start("ctrl", CTRL, f"{C}/grav_comp.yaml")
    assert lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 100, STARTUP), "no controller"

    # arm_joint_meas and controller nominally at 1 kHz. Only check for at
    # least half of it: on a loaded shared machine the real-time sim slows.
    lst.reset_counts()
    lst.spin(2.0)
    rates = {c: lst.count[c] / 2.0 for c in ("franka_arm_joint_meas", "franka_arm_joint_ctrl")}
    print("rates (Hz):", rates)
    assert min(rates.values()) > 500, rates
    # floats where it is (plant gravity comp + damping only)
    assert np.max(np.abs(arm_q(lst) - HOME_Q)) < 0.02

    # gripper: MOVE to 4 cm through the LCM gripper command
    cmd = gripper_cmd_t()
    cmd.timestamp, cmd.seq, cmd.mode = time.time(), 42, GripperCmdData.MOVE
    cmd.width, cmd.speed, cmd.epsilon_inner, cmd.epsilon_outer = 0.04, 0.1, 0.005, 0.005
    lc = lcm.LCM()
    gs = lambda: lst.last.get("franka_gripper_state")  # noqa: E731
    ok = lst.wait_for(lambda: (lc.publish("franka_gripper_cmd", cmd.encode()) or True)
                      and gs() is not None and gs().last_seq == 42 and not gs().busy, 15)
    assert ok, "gripper did not finish the move"
    assert gs().last_success and abs(gs().width - 0.04) < 2e-3


def test_collision_info_and_col_aware_hold(procs, lcm_listener):
    lst = lcm_listener
    procs.start("env", ENV, f"{E}/collision.yaml")
    assert lst.wait_for(lambda: "franka_static_col_info" in lst.last
                        and "franka_robot_col_info" in lst.last, STARTUP)
    static = set(lst.last["franka_static_col_info"].concat_names.split(","))
    robot = set(lst.last["franka_robot_col_info"].concat_names.split(","))
    assert {"floor", "wall_back", "wall_left"} <= static
    assert {"fr3_link5__2", "fr3_hand", "fr3_hand__2"} <= robot

    procs.start("ctrl", CTRL, f"{C}/col_aware_grav_comp.yaml")
    assert lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, STARTUP)
    lst.spin(2.0)
    assert np.max(np.abs(arm_q(lst) - HOME_Q)) < 0.05


def test_impedance_joint_and_task_modes(procs, lcm_listener):
    lst = lcm_listener
    procs.start("env", ENV, f"{E}/default.yaml")
    assert lst.wait_for(_has_arm(lst), STARTUP)
    ctrl = procs.start("ctrl", CTRL, f"{C}/impedance.yaml")
    assert lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, STARTUP)
    for mode in ("j", "t"):
        ctrl.key(mode)
        lst.spin(2.0)
        m = lst.last["franka_arm_joint_meas"]
        assert np.max(np.abs(arm_q(lst) - HOME_Q)) < 0.05, mode
        assert np.max(np.abs(m.qd)) < 0.05, mode
