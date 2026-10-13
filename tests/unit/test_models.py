"""Robot models: MuJoCo MJCF vs URDF (Pinocchio), and the plant's gravity
compensation (the FR3 firmware behaviour the whole repo relies on)."""

import mujoco as mj
import numpy as np
import pinocchio as pin
import pytest

from robot_platform.sim.franka_fr3.SimFrankaFR3 import (
    ARM_JOINT_NAMES,
    FINGER_JOINT_NAMES,
    FR3_HOME_Q,
    SimFrankaFR3,
)

SCENE = "assets/scene/franka_fr3/scene_default.xml"
URDF = "assets/scene/franka_fr3/urdf/fr3_hand.urdf"


def build(gravity_comp=True):
    """Compile the scene the way the env does and return (platform, model, data)."""
    plat = SimFrankaFR3(gravity_comp=gravity_comp)
    spec = mj.MjSpec.from_file(SCENE)
    plat.modify_mjspec(spec)
    model = spec.compile()
    data = mj.MjData(model)
    plat.set_mj_data_name_idx(data)
    plat.reset(data)
    mj.mj_forward(model, data)
    return plat, model, data


def test_mjcf_has_arm_and_fingers():
    model = mj.MjModel.from_xml_path(SCENE)
    for n in ARM_JOINT_NAMES + FINGER_JOINT_NAMES:
        assert model.joint(n) is not None
    assert model.site("fr3_hand_tcp") is not None


def test_urdf_has_9_joints():
    m = pin.buildModelFromUrdf(URDF)
    assert m.nq == 9  # 7 arm + 2 fingers: controllers use num_ctrl_joints 7


@pytest.mark.parametrize("seed", range(5))
def test_fk_mjcf_matches_urdf(seed):
    """TCP pose from MuJoCo and Pinocchio agree at random arm poses."""
    _, model, data = build()
    pm = pin.buildModelFromUrdf(URDF)
    pd = pm.createData()
    rng = np.random.default_rng(seed)
    lo, hi = pm.lowerPositionLimit[:7], pm.upperPositionLimit[:7]
    q = np.r_[rng.uniform(lo, hi), 0.04, 0.04]

    for i, n in enumerate(ARM_JOINT_NAMES):
        data.qpos[model.joint(n).qposadr[0]] = q[i]
    for n in FINGER_JOINT_NAMES:
        data.qpos[model.joint(n).qposadr[0]] = 0.04
    mj.mj_kinematics(model, data)
    base = data.body("base")
    T_mj = np.eye(4)
    site = data.site("fr3_hand_tcp")
    T_mj[:3, :3], T_mj[:3, 3] = site.xmat.reshape(3, 3), site.xpos
    T_base = np.eye(4)
    T_base[:3, :3], T_base[:3, 3] = base.xmat.reshape(3, 3), base.xpos
    T_mj = np.linalg.inv(T_base) @ T_mj

    pin.forwardKinematics(pm, pd, q)
    pin.updateFramePlacements(pm, pd)
    T_pin = pd.oMf[pm.getFrameId("fr3_hand_tcp")].homogeneous
    assert np.allclose(T_mj[:3, 3], T_pin[:3, 3], atol=1e-3)
    assert np.allclose(T_mj[:3, :3], T_pin[:3, :3], atol=1e-3)


def zero_torque_drift(gravity_comp, seconds=1.0):
    plat, model, data = build(gravity_comp)
    idx = plat._arm_qpos_idx
    q0 = data.qpos[idx].copy()
    for _ in range(int(seconds / model.opt.timestep)):
        data.ctrl[plat._arm_act_idx] = 0.0
        mj.mj_step(model, data)
    return np.max(np.abs(data.qpos[idx] - q0))


def test_plant_gravity_comp_holds_arm():
    """gravity_comp: true = the FR3 firmware: zero torque, the arm floats."""
    assert zero_torque_drift(True) < 0.01


def test_without_gravity_comp_arm_falls():
    assert zero_torque_drift(False) > 0.1


def test_link_side_torque_at_rest_is_gravity_load():
    """tau_J (what the FR3 torque sensors measure) carries the gravity load,
    which is what makes payload-mass estimation possible (tutorial 12)."""
    plat, model, data = build(True)
    kp = np.array([600, 600, 600, 600, 250, 150, 50.0])
    kd = np.array([50, 50, 50, 20, 20, 20, 10.0])
    q_des = np.array(FR3_HOME_Q)
    for _ in range(1000):
        q = data.qpos[plat._arm_qpos_idx]
        qd = data.qvel[plat._arm_dof_idx]
        data.ctrl[plat._arm_act_idx] = kp * (q_des - q) - kd * qd
        mj.mj_step(model, data)
    mj.mj_forward(model, data)
    tau_J = plat._link_side_torque(data)
    g = data.qfrc_bias[plat._arm_dof_idx]  # at rest, bias = gravity
    assert np.allclose(tau_J, g, atol=0.3), (tau_J, g)
