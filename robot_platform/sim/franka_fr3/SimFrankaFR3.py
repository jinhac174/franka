"""MuJoCo platform for the Franka Research 3 arm + Franka Hand."""

from __future__ import annotations

import copy
from queue import Queue

import mujoco as mj
import numpy as np

from data_type.basic_types.GripperCmdData import GripperCmdData
from data_type.basic_types.GripperStateData import GripperStateData
from data_type.basic_types.JointCtrlData import JointCtrlData
from data_type.basic_types.JointMeasData import JointMeasData
from robot_platform.sim.BaseSimPlatform import BaseSimPlatform

NUM_ARM_JOINTS = 7
NUM_FINGER_JOINTS = 2

ARM_JOINT_NAMES = [f"fr3_joint{i + 1}" for i in range(NUM_ARM_JOINTS)]
FINGER_JOINT_NAMES = ["fr3_finger_joint1", "fr3_finger_joint2"]
ARM_ACTUATOR_NAMES = [f"motor_{n}" for n in ARM_JOINT_NAMES]
GRIPPER_ACTUATOR_NAME = "actuator_gripper"
GRIPPER_TENDON_NAME = "gripper_width"

# Planning collision model: Franka's own self-collision capsules
# (franka_description 2.9.0, xacro with_sc:=true safety_distance:=0), one row
# per capsule: (name, radius, segment length, centre, axis), expressed in the
# link frame. Extra capsules on a link are named "<link>__<k>" (see
# utils.fcl.getter.col_frame_name). Coarser than the meshes on purpose: fast
# FCL distance queries for the controllers. The union covers every arm-link
# collision-mesh vertex at any pose; see FR3_HAND_BOX for the hand.
FR3_COL_CAPSULES = [
    ("fr3_link0", 0.09, 0.03, (-0.075, 0.0, 0.06), (1.0, 0.0, 0.0)),
    ("fr3_link1", 0.09, 0.283, (0.0, 0.0, -0.1915), (0.0, 0.0, 1.0)),
    ("fr3_link2", 0.09, 0.12, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("fr3_link3", 0.09, 0.15, (0.0, 0.0, -0.145), (0.0, 0.0, 1.0)),
    ("fr3_link4", 0.09, 0.12, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("fr3_link5", 0.09, 0.1, (0.0, 0.0, -0.26), (0.0, 0.0, 1.0)),
    ("fr3_link5__2", 0.055, 0.14, (0.0, 0.08, -0.13), (0.0, 0.0, 1.0)),
    ("fr3_link6", 0.08, 0.08, (0.0, 0.0, -0.03), (0.0, 0.0, 1.0)),
    ("fr3_link7", 0.07, 0.14, (0.0, 0.0, 0.01), (0.0, 0.0, 1.0)),
    ("fr3_link7__2", 0.06, 0.01, (0.0424, 0.0424, 0.082), (0.7071, 0.7071, 0.0)),
    # Finger region (Franka's second hand capsule).
    ("fr3_hand__2", 0.02, 0.1, (0.0, 0.0, 0.1), (0.0, 1.0, 0.0)),
]
# Hand body: Franka's own hand capsule leaves the square housing corners up to
# 33 mm outside (it relies on a 3 cm safety inflation), which matters when
# grasping near a table. Use a box fitted to the hand collision mesh instead:
# (name, full extents, centre) in the fr3_hand frame.
FR3_HAND_BOX = ("fr3_hand", (0.065, 0.21, 0.094), (0.0, -0.002, 0.02))


def _align_z_to(axis) -> np.ndarray:
    """Rotation whose +z points along ``axis`` (FCL capsules lie along z)."""
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, u)
    c = float(z @ u)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def fr3_col_primitives():
    """FR3 + hand planning primitives as ColInfoData.add_col_geom kwargs."""
    prims = []
    for name, radius, length, centre, axis in FR3_COL_CAPSULES:
        offset = np.eye(4)
        offset[:3, :3] = _align_z_to(axis)
        offset[:3, 3] = centre
        prims.append(
            {
                "name": name,
                "geom_type": "capsule",
                "size": np.array([radius, length]),
                "offset": offset,
            }
        )
    name, extents, centre = FR3_HAND_BOX
    offset = np.eye(4)
    offset[:3, 3] = centre
    prims.append(
        {
            "name": name,
            "geom_type": "box",
            "size": np.array(extents),
            "offset": offset,
        }
    )
    return prims


# Franka's standard "ready" pose (libfranka / franka_ros examples).
FR3_HOME_Q = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
GRIPPER_MAX_WIDTH = 0.08


class _GripperSim:
    """Emulates the Franka Hand's blocking move/grasp API.

    The real gripper is not torque controlled: libfranka exposes
    move(width, speed), grasp(width, speed, force, eps_in, eps_out), stop() and
    homing(), each of which runs to completion on the gripper's own controller.
    This class reproduces that on a MuJoCo position servo: each sim step it
    returns a width set-point (ramped at the commanded speed) and a force cap.
    A grasp drives the set-point fully closed, so contact -- not the set-point
    -- stops the fingers, and the servo then saturates at the grasp force.
    """

    MOVE_FORCE = 70.0  # force cap while moving / holding (N)
    # |width rate| below this counts as stopped (m/s). Soft contacts under a
    # strong squeeze keep creeping a few mm/s, so this cannot be ~0.
    STALL_SPEED = 5e-3
    STALL_TIME = 0.1  # stopped this long -> grasp finished (s)
    MOVE_TIMEOUT = 0.5  # stalled this long short of a move target -> blocked

    def __init__(self, width0: float):
        self.max_width = GRIPPER_MAX_WIDTH
        self.reset(width0)

    def reset(self, width0: float):
        self.mode = GripperCmdData.STOP
        self.busy = False
        self.width_ref = width0
        self.goal = width0
        self.target = width0
        self.speed = 0.1
        self.force = self.MOVE_FORCE
        self.eps_in = 0.005
        self.eps_out = 0.005
        self.grasp_held = False
        self.stall_t = 0.0
        self.last_seq = -1
        self.last_success = True
        self._pending_seq = -1

    def command(self, cmd: GripperCmdData, width: float):
        if cmd.seq == self._pending_seq or cmd.seq == self.last_seq:
            return  # already executing / executed
        self._pending_seq = cmd.seq
        self.stall_t = 0.0
        self.grasp_held = False
        self.width_ref = width

        mode = cmd.mode
        if mode == GripperCmdData.STOP:
            self.mode = GripperCmdData.STOP
            self.goal = width
            self.force = self.MOVE_FORCE
            self._finish(True)
            return
        if mode == GripperCmdData.HOMING:
            mode, cmd_width, speed = GripperCmdData.MOVE, self.max_width, 0.1
        else:
            cmd_width, speed = cmd.width, cmd.speed

        self.mode = mode
        self.target = float(np.clip(cmd_width, 0.0, self.max_width))
        self.speed = max(float(speed), 1e-3)
        self.eps_in = cmd.epsilon_inner
        self.eps_out = cmd.epsilon_outer
        if mode == GripperCmdData.GRASP:
            # Close (or open) all the way; the object stops the fingers.
            self.goal = 0.0 if self.target <= width else self.max_width
            self.force = float(np.clip(cmd.force, 0.0, 140.0))
        else:
            self.goal = self.target
            self.force = self.MOVE_FORCE
        self.busy = True

    def _finish(self, success: bool):
        self.busy = False
        self.last_seq = self._pending_seq
        self.last_success = success

    def is_grasped(self, width: float) -> bool:
        return (
            self.grasp_held
            and self.target - self.eps_in <= width <= self.target + self.eps_out
        )

    def step(self, width: float, width_rate: float, dt: float):
        """Advance one sim step. Returns (width set-point [m], force cap [N])."""
        step = self.speed * dt
        self.width_ref += float(np.clip(self.goal - self.width_ref, -step, step))

        if self.busy:
            if abs(width_rate) < self.STALL_SPEED:
                self.stall_t += dt
            else:
                self.stall_t = 0.0
            ref_done = abs(self.width_ref - self.goal) < 1e-9

            if self.mode == GripperCmdData.MOVE:
                if ref_done and abs(width - self.target) < 1e-3:
                    self._finish(True)
                elif self.stall_t > self.MOVE_TIMEOUT:  # blocked by something
                    self.goal = self.width_ref = width
                    self._finish(False)
            elif self.mode == GripperCmdData.GRASP and self.stall_t > self.STALL_TIME:
                success = (
                    self.target - self.eps_in <= width <= self.target + self.eps_out
                )
                self.grasp_held = success
                if not success:  # nothing (or the wrong thing) in the hand
                    self.mode = GripperCmdData.STOP
                    self.goal = self.width_ref = width
                    self.force = self.MOVE_FORCE
                self._finish(success)

        return self.width_ref, self.force


class SimFrankaFR3(BaseSimPlatform):
    """MuJoCo platform for the 7-DoF Franka Research 3 + Franka Hand.

    Arm (``ctrl_channel``, joint_ctrl_t with 7 joints)::

        tau = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff

    ``gravity_comp: true`` (default) mimics the FR3 firmware, which always
    compensates gravity itself in torque control: the commanded torque is
    *on top of* gravity, so controllers must NOT add g(q). Implemented with
    MuJoCo body gravcomp (gravity only, no Coriolis -- same as the robot).
    ``gravity_comp: false`` turns it off so a controller can do it (tutorial 03).

    Gripper (``gripper_cmd_channel``, gripper_cmd_t): move / grasp / stop /
    homing, emulated by ``_GripperSim``.

    Published:
      * ``arm_joint_meas_channel``     joint_meas_t (7)  q, qd, tau
        (tau = link-side joint torque, like the FR3 torque sensors / libfranka
        tau_J: motor torque incl. gravity comp, minus gear damping and
        friction. See ``_link_side_torque``.)
      * ``gripper_joint_meas_channel`` joint_meas_t (2)  finger positions
        (each = width / 2), so URDF/Pinocchio models see [arm(7), fingers(2)]
      * ``gripper_state_channel``      gripper_state_t
    """

    def __init__(
        self,
        *args,
        default_q_values=None,
        default_gripper_width: float = GRIPPER_MAX_WIDTH,
        gravity_comp: bool = True,
        arm_joint_meas_freq: int = 1000,
        gripper_meas_freq: int = 100,
        arm_joint_meas_channel: str = "franka_arm_joint_meas",
        gripper_joint_meas_channel: str = "franka_gripper_joint_meas",
        gripper_state_channel: str = "franka_gripper_state",
        gripper_cmd_channel: str = "franka_gripper_cmd",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if default_q_values is None:
            default_q_values = FR3_HOME_Q
        if len(default_q_values) != NUM_ARM_JOINTS:
            raise ValueError(
                f"default_q_values must have {NUM_ARM_JOINTS} entries, "
                f"got {len(default_q_values)}."
            )
        self._default_q = np.array(default_q_values, dtype=np.float64)
        self._default_width = float(default_gripper_width)
        self._gravity_comp = bool(gravity_comp)

        self.arm_joint_meas_data = JointMeasData(num_joints=NUM_ARM_JOINTS)
        self.gripper_joint_meas_data = JointMeasData(num_joints=NUM_FINGER_JOINTS)
        self.gripper_state_data = GripperStateData()
        self._last_ctrl_data = JointCtrlData(num_joints=NUM_ARM_JOINTS)
        self._ever_received_ctrl = False

        self._gripper = _GripperSim(self._default_width)
        self._gripper_cmd_channel = gripper_cmd_channel
        self._gripper_cmd_que: Queue | None = None

        self._arm_ch = arm_joint_meas_channel
        self._finger_ch = gripper_joint_meas_channel
        self._gstate_ch = gripper_state_channel
        self._pub_dt = {
            self._arm_ch: 1.0 / arm_joint_meas_freq,
            self._finger_ch: 1.0 / gripper_meas_freq,
            self._gstate_ch: 1.0 / gripper_meas_freq,
        }
        self._last_pub_time = {ch: 0.0 for ch in self._pub_dt}

        self._arm_qpos_idx = []
        self._arm_dof_idx = []
        self._finger_qpos_idx = []
        self._finger_dof_idx = []
        self._arm_act_idx = []
        self._gripper_act_idx = -1
        self._gripper_tendon_idx = -1

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def modify_mjspec(self, mj_spec: mj.MjSpec):
        if not self._gravity_comp:
            return

        def _set(body):
            body.gravcomp = 1.0
            for child in body.bodies:
                _set(child)

        _set(mj_spec.body("base"))

    def set_mj_data_name_idx(self, mj_data: mj.MjData):
        model = mj_data.model
        for name in ARM_JOINT_NAMES:
            j = model.joint(name)
            self._arm_qpos_idx.append(int(j.qposadr[0]))
            self._arm_dof_idx.append(int(j.dofadr[0]))
        for name in FINGER_JOINT_NAMES:
            j = model.joint(name)
            self._finger_qpos_idx.append(int(j.qposadr[0]))
            self._finger_dof_idx.append(int(j.dofadr[0]))
        self._arm_act_idx = [model.actuator(n).id for n in ARM_ACTUATOR_NAMES]
        self._gripper_act_idx = model.actuator(GRIPPER_ACTUATOR_NAME).id
        self._gripper_tendon_idx = model.tendon(GRIPPER_TENDON_NAME).id

    def set_sub_que_dict(self, que_dict: dict):
        self._gripper_cmd_que = que_dict.get("intr_sub_que_dict", {}).get(
            self._gripper_cmd_channel
        )

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def apply_sim_control(self, ctrl_data: JointCtrlData, mj_data: mj.MjData):
        # Zero-order hold: with no new command, re-apply the last one.
        if ctrl_data is None:
            ctrl_data = self._last_ctrl_data
        else:
            self._ever_received_ctrl = True

        q = mj_data.qpos[self._arm_qpos_idx]
        qd = mj_data.qvel[self._arm_dof_idx]
        t, q_des, qd_des, tau_ff, kp, kd = ctrl_data.get_data()
        self._last_ctrl_data.set_data(t, q_des, qd_des, tau_ff, kp, kd)

        tau = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff
        if not self._gravity_comp and not self._ever_received_ctrl:
            # Startup-only hold when the plant has no gravity comp: without it
            # the arm would collapse before the controller connects. Once a
            # controller is driving, gravity is its job (no double comp).
            tau = tau + mj_data.qfrc_bias[self._arm_dof_idx]
        mj_data.ctrl[self._arm_act_idx] = tau

        self._apply_gripper_control(mj_data)

    def _apply_gripper_control(self, mj_data: mj.MjData):
        width = float(mj_data.ten_length[self._gripper_tendon_idx])
        if self._gripper_cmd_que is not None:
            while not self._gripper_cmd_que.empty():
                self._gripper.command(self._gripper_cmd_que.get_nowait(), width)
        width_rate = float(mj_data.ten_velocity[self._gripper_tendon_idx])
        model = mj_data.model
        width_ref, force_cap = self._gripper.step(
            width, width_rate, float(model.opt.timestep)
        )
        mj_data.ctrl[self._gripper_act_idx] = width_ref
        model.actuator_forcerange[self._gripper_act_idx] = (-force_cap, force_cap)

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------

    def _due(self, ch: str, t: float) -> bool:
        # Accumulator deadline so the long-run publish rate matches pub_dt.
        if t + 1e-9 >= self._last_pub_time[ch] + self._pub_dt[ch]:
            self._last_pub_time[ch] += self._pub_dt[ch]
            return True
        return False

    def _link_side_torque(self, mj_data: mj.MjData) -> np.ndarray:
        """What the FR3's joint torque sensors measure (tau_J).

        The sensors sit after the gearbox, so they see the motor torque minus
        the gear's damping and dry friction. In MuJoCo terms: actuator force +
        passive force (gravity comp + joint damping; qfrc_passive already
        includes qfrc_gravcomp) + the joint-friction constraint force. Because
        friction is removed, tau_J at rest equals the load on the joint, which
        is what makes external-force (and object-mass) estimation possible.
        """
        idx = self._arm_dof_idx
        tau = mj_data.qfrc_actuator[idx] + mj_data.qfrc_passive[idx]
        friction = np.zeros(mj_data.model.nv)
        for r in np.flatnonzero(mj_data.efc_type == mj.mjtConstraint.mjCNSTR_FRICTION_DOF):
            friction[mj_data.efc_id[r]] += mj_data.efc_force[r]
        return tau + friction[idx]

    def sync_intr_data_from_sim(self, mj_data: mj.MjData, intr_pub_que_dict: dict):
        t = mj_data.time

        if self._due(self._arm_ch, t):
            idx = self._arm_dof_idx
            tau = self._link_side_torque(mj_data)
            self.arm_joint_meas_data.set_data(
                t, mj_data.qpos[self._arm_qpos_idx], mj_data.qvel[idx], tau
            )
            intr_pub_que_dict[self._arm_ch].put(
                copy.deepcopy(self.arm_joint_meas_data)
            )

        if self._due(self._finger_ch, t):
            idx = self._finger_dof_idx
            self.gripper_joint_meas_data.set_data(
                t,
                mj_data.qpos[self._finger_qpos_idx],
                mj_data.qvel[idx],
                mj_data.qfrc_actuator[idx],
            )
            intr_pub_que_dict[self._finger_ch].put(
                copy.deepcopy(self.gripper_joint_meas_data)
            )

        if self._gstate_ch in intr_pub_que_dict and self._due(self._gstate_ch, t):
            width = float(mj_data.ten_length[self._gripper_tendon_idx])
            g = self._gripper
            self.gripper_state_data.set_data(
                t,
                width,
                g.max_width,
                g.is_grasped(width),
                g.busy,
                g.last_seq,
                g.last_success,
            )
            intr_pub_que_dict[self._gstate_ch].put(
                copy.deepcopy(self.gripper_state_data)
            )

    # ------------------------------------------------------------------
    # Reset / collision info
    # ------------------------------------------------------------------

    def reset(self, mj_data: mj.MjData):
        # Seed the held command with a light PD at the default pose, so the
        # arm stays put if the env runs before/without a controller.
        self._ever_received_ctrl = False
        self._last_ctrl_data.set_data(
            mj_data.time,
            self._default_q,
            np.zeros(NUM_ARM_JOINTS),
            np.zeros(NUM_ARM_JOINTS),
            np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]),
            np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0]),
        )
        self.arm_joint_meas_data.set_q(self._default_q)
        half = self._default_width / 2.0
        self.gripper_joint_meas_data.set_q(np.array([half, half]))

        mj_data.qpos[self._arm_qpos_idx] = self._default_q
        mj_data.qpos[self._finger_qpos_idx] = half
        mj_data.qvel[self._arm_dof_idx] = 0.0
        mj_data.qvel[self._finger_dof_idx] = 0.0
        mj_data.ctrl[:] = 0.0
        mj_data.ctrl[self._gripper_act_idx] = self._default_width
        self._gripper.reset(self._default_width)

        for ch in self._last_pub_time:
            self._last_pub_time[ch] = mj_data.time

    def get_col_info_data(self):
        """Robot planning primitives (capsules) in each link's local frame.
        Published by the collision env on robot_col_info; controllers place
        them with FK and run FCL distance queries."""
        return fr3_col_primitives()
