"""Parallel-jaw grasp quality: predicted vs simulated minimum grip force.

Tutorial 11. A floating Robotiq 2F-85 (no arm) grasps an object at sampled
antipodal grasps. For each grasp the script compares

* the grip force predicted by the soft-finger model
  (``utils.grasping.parallel_jaw.required_grip_force``), with
* the smallest force for which the object survives "gravity on + lift" in
  MuJoCo (found by bisection).

    python run_grasp_quality.py --config configs/grasp_quality/default.yaml
    python run_grasp_quality.py --config ... --view     # watch it in MuJoCo
"""

from __future__ import annotations

import argparse
import copy
import os
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from data_type.basic_types.GripperCmdData import GripperCmdData
from robot_platform.gripper import robotiq_2f85 as rq
from robot_platform.sim.franka_fr3.SimFrankaFR3 import GRIPPER_MAX_WIDTH, _GripperSim
from utils.grasping.objects import GraspObject
from utils.grasping.parallel_jaw import (
    TCP_OFFSET,
    HandCollisionChecker,
    grasp_tcp_poses,
    required_grip_force,
    sample_antipodal_grasps,
)

ROBOT_XML = "assets/scene/franka_fr3/fr3_robotiq_robot.xml"


def build_hand_only_xml(robot_xml: str = ROBOT_XML) -> str:
    """MJCF string: a free-floating Robotiq 2F-85 from ``robot_xml`` that a
    weld constraint pulls toward a mocap target.

    Copies the gripper's defaults, assets, body subtree, contact excludes,
    tendon, equality constraints and actuator, so it is identical to the one
    on the FR3.

    Why not put the gripper directly on the mocap body? Moving a mocap body
    teleports it: it gets a new position but zero velocity, and MuJoCo's
    friction acts on velocities, so a lifted object would just stay behind.
    The weld moves the gripper dynamically, with real velocities.
    """
    base = os.path.dirname(os.path.abspath(robot_xml))
    src = ET.parse(robot_xml).getroot()
    hand = next(b for b in src.iter("body") if b.get("name") == "fr3_gripper")
    hand = copy.deepcopy(hand)
    hand.set("pos", "0 0 0")
    hand.set("quat", "1 0 0 0")
    bodies = {b.get("name") for b in hand.iter("body")}

    root = ET.Element("mujoco", model="robotiq_2f85_only")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    # Elliptic friction cones match the model's friction cone. MuJoCo's
    # contacts are soft: under a steady load a held object slowly "creeps"
    # even far inside the friction cone. The noslip post-solver removes that
    # creep, so slipping in this sim means the friction limit was exceeded.
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast",
                  cone="elliptic", impratio="10", noslip_iterations="5")
    default = ET.SubElement(root, "default")
    for d in src.find("default").findall("default"):
        if d.get("class") == "robotiq":
            default.append(copy.deepcopy(d))
    asset = ET.SubElement(root, "asset")
    for el in src.find("asset"):
        if el.tag == "material" and el.get("name", "").startswith("robotiq_"):
            asset.append(copy.deepcopy(el))
        if el.tag == "mesh" and "meshes/robotiq/" in el.get("file", ""):
            m = copy.deepcopy(el)
            m.set("file", os.path.join(base, el.get("file")))
            asset.append(m)
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 1.5", dir="0 0 -1", directional="true")
    ET.SubElement(world, "body", name="hand_target", mocap="true")
    ET.SubElement(hand, "freejoint", name="hand_free")
    world.append(hand)
    contact = ET.SubElement(root, "contact")
    for ex in src.find("contact"):
        if ex.get("body1") in bodies and ex.get("body2") in bodies:
            contact.append(copy.deepcopy(ex))
    root.append(copy.deepcopy(src.find("tendon")))
    equality = copy.deepcopy(src.find("equality"))
    ET.SubElement(equality, "weld", body1="hand_target", body2="fr3_gripper",
                  solref="0.005 1")
    root.append(equality)
    act = ET.SubElement(root, "actuator")
    for el in src.find("actuator"):
        if el.get("name") == "actuator_gripper":
            act.append(copy.deepcopy(el))
    return ET.tostring(root, encoding="unicode")


class HandOnlySim:
    """Floating Robotiq 2F-85 + object, with the gripper's move/grasp API."""

    def __init__(self, obj: GraspObject):
        spec = mj.MjSpec.from_string(build_hand_only_xml())
        obj.add_to_spec(spec)
        self.m = spec.compile()
        self.d = mj.MjData(self.m)
        self.obj = obj
        self.body = self.m.body("grasp_object").id
        self.qadr = int(self.m.jnt_qposadr[self.m.body_jntadr[self.body]])
        self.dadr = int(self.m.jnt_dofadr[self.m.body_jntadr[self.body]])
        self.act = self.m.actuator("actuator_gripper").id
        self.hand = self.m.body("fr3_gripper").id
        self.hand_qadr = int(self.m.joint("hand_free").qposadr[0])
        self.gripper = _GripperSim(GRIPPER_MAX_WIDTH)
        self._seq = 0
        self._prev_width = 0.0

    @property
    def width(self) -> float:
        return rq.mj_width(self.m, self.d)

    def set_gravity(self, on: bool):
        self.m.opt.gravity[:] = (0.0, 0.0, -9.81) if on else (0.0, 0.0, 0.0)

    def reset(self, T_world_obj: np.ndarray, T_world_tcp: np.ndarray, open_width: float):
        mj.mj_resetData(self.m, self.d)
        self.set_gravity(False)
        q = Rotation.from_matrix(T_world_obj[:3, :3]).as_quat()
        self.d.qpos[self.qadr : self.qadr + 3] = T_world_obj[:3, 3]
        self.d.qpos[self.qadr + 3 : self.qadr + 7] = [q[3], q[0], q[1], q[2]]
        rq.mj_set_width(self.m, self.d, open_width)
        self.set_hand(T_world_tcp)
        # Start the hand exactly at its target.
        self.d.qpos[self.hand_qadr : self.hand_qadr + 3] = self.d.mocap_pos[0]
        self.d.qpos[self.hand_qadr + 3 : self.hand_qadr + 7] = self.d.mocap_quat[0]
        self.gripper.reset(open_width)
        mj.mj_forward(self.m, self.d)
        self._prev_width = self.width
        rq.mj_actuate(self.m, self.d, self.act, open_width, _GripperSim.MOVE_FORCE, open_width)

    def set_hand(self, T_world_tcp: np.ndarray):
        # Hand frame = TCP frame moved back along the approach axis.
        R = T_world_tcp[:3, :3]
        p = T_world_tcp[:3, 3] - TCP_OFFSET * R[:, 2]
        q = Rotation.from_matrix(R).as_quat()
        self.d.mocap_pos[0] = p
        self.d.mocap_quat[0] = [q[3], q[0], q[1], q[2]]

    def command(self, mode, width=0.0, speed=0.1, force=0.0, eps=0.01):
        self._seq += 1
        c = GripperCmdData()
        c.set_data(self.d.time, self._seq, mode, width, speed, force, eps, eps)
        self.gripper.command(c, self.width)
        if mode == GripperCmdData.GRASP:
            # Analysis: also below the 2F-85's 20 N minimum, to find the
            # physical minimum force (the real gripper cannot go that low).
            self.gripper.force = float(force)

    def step(self, n: int = 1):
        dt = self.m.opt.timestep
        for _ in range(n):
            w = self.width
            ref, cap = self.gripper.step(w, (w - self._prev_width) / dt, dt)
            self._prev_width = w
            rq.mj_actuate(self.m, self.d, self.act, ref, cap, w)
            mj.mj_step(self.m, self.d)

    def object_in_hand(self) -> np.ndarray:
        """Object pose relative to the hand (to measure slip)."""
        T_h = np.eye(4)
        T_h[:3, :3] = self.d.xmat[self.hand].reshape(3, 3)
        T_h[:3, 3] = self.d.xpos[self.hand]
        T_o = np.eye(4)
        T_o[:3, :3] = self.d.xmat[self.body].reshape(3, 3)
        T_o[:3, 3] = self.d.xpos[self.body]
        return np.linalg.inv(T_h) @ T_o


def trial(sim: HandOnlySim, T_world_obj, T_world_tcp, width, force, cfg, viewer=None):
    """Grasp with ``force`` (gravity off), then gravity on + lift.

    Returns ``(held, slip_mm, slip_deg, grasp_width)``.
    """
    open_w = min(width + cfg.open_margin, GRIPPER_MAX_WIDTH)
    sim.reset(T_world_obj, T_world_tcp, open_w)

    def step():
        sim.step()
        if viewer is not None and int(round(sim.d.time * 1000)) % 16 == 0:
            viewer.sync()

    def run(seconds):
        for _ in range(int(round(seconds / sim.m.opt.timestep))):
            step()

    sim.command(GripperCmdData.GRASP, width, cfg.grasp_speed, force, cfg.grasp_eps)
    run(cfg.close_time)
    if not sim.gripper.last_success:
        return False, np.inf, np.inf, sim.width
    T0 = sim.object_in_hand()

    sim.set_gravity(True)
    run(cfg.hold_time)
    # Lift: 0.1 m up in 0.5 s (peak acceleration ~2.4 m/s^2), then hold.
    p0 = sim.d.mocap_pos[0].copy()
    steps = int(round(cfg.lift_time / sim.m.opt.timestep))
    for k in range(steps):
        s = (k + 1) / steps
        sim.d.mocap_pos[0] = p0 + np.array([0, 0, cfg.lift_height * (10 * s**3 - 15 * s**4 + 6 * s**5)])
        step()
    run(cfg.hold_time)

    dT = np.linalg.inv(T0) @ sim.object_in_hand()
    slip_mm = 1000.0 * float(np.linalg.norm(dT[:3, 3]))
    slip_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(dT[:3, :3]).as_rotvec())))
    held = slip_mm < cfg.slip_tol_mm and slip_deg < cfg.slip_tol_deg
    return held, slip_mm, slip_deg, sim.width


def select_grasps(obj: GraspObject, cfg, T_world_obj):
    """Collision-free candidate TCP poses (world frame), a spread of COM offsets."""
    g = sample_antipodal_grasps(obj.grasp_mesh, num_samples=cfg.num_samples, mu=cfg.mu, seed=cfg.seed)
    T_obj_tcp, idx = grasp_tcp_poses(g, cfg.num_approach)
    chk = HandCollisionChecker(obj.mesh)
    out = []
    for T, i in zip(T_obj_tcp, idx):
        width = float(g.width[i])
        if chk.collides(T, min(width + cfg.open_margin, GRIPPER_MAX_WIDTH), margin=0.002):
            continue
        T_w = T_world_obj @ T
        # Keep grasps whose approach points down at least a little (as an arm
        # reaching an object on a table would).
        if T_w[2, 2] > -cfg.min_downward:
            continue
        R_wo = T_world_obj[:3, :3]
        out.append(
            (T_w, width, float(np.degrees(g.cone_angle[i])), R_wo @ g.n1[i], R_wo @ g.n2[i])
        )
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(out)
    return out[: cfg.num_grasps]


def main():
    parser = argparse.ArgumentParser(description="Parallel-jaw grasp quality (tutorial 11)")
    parser.add_argument("--config", default="configs/grasp_quality/default.yaml")
    parser.add_argument("--view", action="store_true", help="show the trials in MuJoCo")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    obj = GraspObject(cfg.object_config)
    sim = HandOnlySim(obj)
    T_world_obj = np.eye(4)
    T_world_obj[:3, :3] = obj.R_spawn
    com_world = T_world_obj[:3, :3] @ obj.com_cad()

    grasps = select_grasps(obj, cfg, T_world_obj)
    print(f"[GraspQuality] {obj.name}: mass {obj.mass} kg, mu {cfg.mu}, gamma {cfg.gamma} m; "
          f"{len(grasps)} grasps")

    viewer = None
    if args.view:
        import mujoco.viewer as mjv

        viewer = mjv.launch_passive(sim.m, sim.d)

    print("F_square: model assuming the surfaces face the fingers squarely")
    print("F_pred:   model with each contact's friction cone around its real normal")
    print(f"{'#':>2} {'width':>6} {'tilt':>5} {'r_com':>6} {'tau_axis':>8} | "
          f"{'F_square':>8} {'F_pred':>6} {'F_sim':>6}  sim/pred")
    rows = []
    for k, (T_w, width, cone_deg, n1, n2) in enumerate(grasps):
        F_sq, _ = required_grip_force(T_w, width, com_world, obj.mass, cfg.mu, cfg.gamma)
        F_pred, det = required_grip_force(
            T_w, width, com_world, obj.mass, cfg.mu, cfg.gamma, n1_world=n1, n2_world=n2
        )
        r_com = 1000.0 * np.linalg.norm(com_world - T_w[:3, 3])
        # The 2F-85 model's springs already squeeze ~2 N with the motor off:
        # the sim cannot test less than that.
        lo, hi = max(cfg.f_min, rq.sim_min_squeeze(width)), cfg.f_max
        if not trial(sim, T_world_obj, T_w, width, hi, cfg, viewer)[0]:
            print(f"{k:2d} {width:6.3f} {cone_deg:5.1f} {r_com:5.0f}mm {det['tau_axis']:7.3f}Nm | "
                  f"{F_sq:8.1f} {F_pred:6.1f}   >{hi:.0f}  (not held even at max force)")
            continue
        if trial(sim, T_world_obj, T_w, width, lo, cfg, viewer)[0]:
            print(f"{k:2d} {width:6.3f} {cone_deg:5.1f} {r_com:5.0f}mm {det['tau_axis']:7.3f}Nm | "
                  f"{F_sq:8.1f} {F_pred:6.1f}  <={lo:3.1f}  (held at the sim's minimum squeeze)")
            continue
        for _ in range(cfg.bisection_steps):
            mid = 0.5 * (lo + hi)
            if trial(sim, T_world_obj, T_w, width, mid, cfg, viewer)[0]:
                hi = mid
            else:
                lo = mid
        rows.append((F_pred, hi))
        print(f"{k:2d} {width:6.3f} {cone_deg:5.1f} {r_com:5.0f}mm {det['tau_axis']:7.3f}Nm | "
              f"{F_sq:8.1f} {F_pred:6.1f} {hi:6.1f}  {hi / max(F_pred, 1e-6):5.2f}")

    if rows:
        F_pred, F_sim = np.array(rows).T
        print(f"[GraspQuality] median F_sim / F_pred = {np.median(F_sim / F_pred):.2f}, "
              f"correlation {np.corrcoef(F_pred, F_sim)[0, 1]:.2f}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
