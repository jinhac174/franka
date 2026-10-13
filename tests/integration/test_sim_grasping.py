"""Tutorials 10-12 end-to-end: reactive parallel-jaw grasp, grasp-quality
model vs simulation, and force-aware (weigh, then re-grasp) grasping."""

import re
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf

from tests.helpers import STARTUP, CTRL, ENV

E = "configs/franka_fr3/env/grasping.yaml"
C = "configs/franka_fr3/controllers"


def _obj_z(lst):
    return float(lst.last["franka_object_pose"].position[2])


def _grasp_and_lift(procs, lst, ctrl_cfg, seed, timeout=300):
    procs.start("env", ENV, E, f"--obj_spawn_seed {seed}")
    assert lst.wait_for(lambda: "franka_object_pose" in lst.last
                        and "franka_arm_joint_meas" in lst.last, STARTUP), "env did not publish"
    lst.spin(1.0)  # object settles on the table
    z0 = _obj_z(lst)
    ctrl = procs.start("ctrl", CTRL, f"{C}/{ctrl_cfg}")
    assert lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, STARTUP), "no controller"
    lst.spin(2.0)
    ctrl.key("v")
    lifted = lst.wait_for(lambda: _obj_z(lst) - z0 > 0.08, timeout)
    assert lifted, f"object not lifted (dz={_obj_z(lst) - z0:.3f})\n{ctrl.log()[-3000:]}"
    # still held a little later, fingers closed on it
    lst.spin(3.0)
    gs = lst.last["franka_gripper_state"]
    assert _obj_z(lst) - z0 > 0.08 and gs.is_grasped
    return ctrl


def test_reactive_parallel_jaw_grasp(procs, lcm_listener):
    """Tutorial 10: sample antipodal grasps, IK reach, approach, close, lift."""
    _grasp_and_lift(procs, lcm_listener, "parallel_jaw_grasp.yaml", seed=2)


def test_force_aware_grasp_weighs_object(procs, lcm_listener):
    """Tutorial 12: probe grasp, weigh with tau_J, re-grasp with the model force."""
    ctrl = _grasp_and_lift(procs, lcm_listener, "force_aware_grasp.yaml", seed=7, timeout=400)
    m = re.findall(r"Weighed: m_est = ([\d.]+) kg \(env says ([\d.]+) kg\)", ctrl.log())
    assert m, ctrl.log()[-3000:]
    m_est, m_true = map(float, m[-1])
    assert abs(m_est - m_true) / m_true < 0.25, (m_est, m_true)


def test_grasp_quality_model_matches_sim(tmp_path):
    """Tutorial 11 (few grasps on the can): predicted vs simulated grip force."""
    cfg = OmegaConf.load("configs/grasp_quality/default.yaml")
    cfg.object_config, cfg.num_grasps = "configs/objects/tomato_soup_can.yaml", 5
    path = tmp_path / "gq.yaml"
    OmegaConf.save(cfg, path)
    out = subprocess.run([sys.executable, "run_grasp_quality.py", "--config", str(path)],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    m = re.search(r"median F_sim / F_pred = ([\d.]+), correlation ([-\d.]+)", out.stdout)
    assert m, out.stdout[-3000:]
    ratio, corr = float(m.group(1)), float(m.group(2))
    assert 0.7 < ratio < 1.6 and corr > 0.8, out.stdout[-2000:]
