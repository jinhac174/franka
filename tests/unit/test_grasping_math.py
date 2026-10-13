"""Grasp geometry / mechanics (utils.grasping) and the perception bridge math."""

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from run_pose_bridge import bb_pose_to_mesh_pose
from utils.grasping.objects import GraspObject
from robot_platform.gripper import robotiq_2f85 as rq
from utils.grasping.parallel_jaw import (
    MAX_WIDTH,
    TCP_OFFSET,
    HandCollisionChecker,
    grasp_tcp_poses,
    hand_boxes,
    required_grip_force,
    sample_antipodal_grasps,
)


@pytest.fixture(scope="module")
def can():
    return GraspObject("configs/objects/tomato_soup_can.yaml")


def test_antipodal_grasps_on_can_are_valid(can):
    g = sample_antipodal_grasps(can.grasp_mesh, num_samples=1500, seed=0)
    assert len(g) > 20
    assert np.all((g.width >= 0.01) & (g.width <= MAX_WIDTH - 0.01))
    assert np.all(g.cone_angle <= 0.6 * np.arctan(0.8) + 1e-9)
    # antipodal: outward normals oppose each other along the closing axis
    assert np.all(np.einsum("ij,ij->i", g.n1, g.axis) < 0)
    assert np.all(np.einsum("ij,ij->i", g.n2, g.axis) > 0)


def test_tcp_poses_put_fingers_on_contacts(can):
    g = sample_antipodal_grasps(can.grasp_mesh, num_samples=800, seed=1)
    T, idx = grasp_tcp_poses(g, num_approach=4)
    assert len(T) == 4 * len(g)
    for Ti, i in zip(T, idx):
        R = Ti[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6) and np.linalg.det(R) > 0
        # the pad centres (on the 2F-85's arc, pad_z beyond the TCP once
        # closed to the pair's width) land midway between the contacts
        pads = Ti[:3, 3] + (rq.pad_z(g.width[i]) - TCP_OFFSET) * R[:, 2]
        assert np.allclose(pads, g.centre[i])
        assert abs(R[:, 1] @ g.axis[i]) > 1 - 1e-9  # y = closing axis


def test_hand_collision_checker(can):
    chk = HandCollisionChecker(can.grasp_mesh)
    centre = can.grasp_mesh.bounds.mean(axis=0)
    T_in = np.eye(4)
    T_in[:3, 3] = centre
    assert not chk.collides(T_in, 0.08)  # open fingers straddle the can
    assert chk.collides(T_in, 0.02)  # nearly closed fingers are inside it
    T_palm = T_in.copy()
    T_palm[2, 3] += 0.1  # palm (behind the TCP) runs into the can
    assert chk.collides(T_palm, 0.08)
    T_far = np.eye(4)
    T_far[:3, 3] = centre + [0, 0, -1.0]
    assert not chk.collides(T_far, 0.08)


def test_required_grip_force_basic_mechanics():
    """Side grasp through the COM: friction carries the weight,
    F = m g / (2 mu). Moving the COM off the axis needs more force."""
    T = np.eye(4)  # y = closing axis, horizontal; gravity along -z (tangential)
    T[:3, 3] = [0.5, 0.0, 0.3]
    m, mu = 0.5, 0.8
    F0, _ = required_grip_force(T, 0.06, T[:3, 3], m, mu, gamma=0.008)
    assert F0 == pytest.approx(m * 9.81 / (2 * mu), rel=0.05)
    F1, _ = required_grip_force(T, 0.06, T[:3, 3] + [0.05, 0, 0], m, mu, gamma=0.008)
    assert F1 > 1.5 * F0
    F2, _ = required_grip_force(T, 0.06, T[:3, 3], m, 2 * mu, gamma=0.008)
    assert F2 < F0


def test_pose_bridge_inverts_foundation_pose_convention():
    """FoundationPose reports T_world_bb = T_world_mesh @ inv(to_origin);
    the bridge must recover T_world_mesh (and apply base_from_world)."""
    mesh = trimesh.load("assets/object_mesh/ycb_006_mustard_bottle/textured_mesh.obj", force="mesh")
    to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
    T_world_mesh = np.eye(4)
    T_world_mesh[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.2, 1.0]).as_matrix()
    T_world_mesh[:3, 3] = [0.5, -0.1, 0.42]
    T_world_bb = T_world_mesh @ np.linalg.inv(to_origin)
    vec = np.r_[T_world_bb[:3, 3], T_world_bb[:3, :3].reshape(-1)]
    assert np.allclose(bb_pose_to_mesh_pose(vec, to_origin), T_world_mesh, atol=1e-9)

    T_base_world = np.eye(4)
    T_base_world[:3, 3] = [0.1, 0.0, -0.02]
    out = bb_pose_to_mesh_pose(vec, to_origin, T_base_world)
    assert np.allclose(out, T_base_world @ T_world_mesh, atol=1e-9)


@pytest.mark.parametrize("angle", [0.0, 0.3, 0.6])
def test_hand_boxes_cover_gripper_meshes(angle):
    """The grasp planner's gripper boxes contain the 2F-85's collision meshes
    (MuJoCo model) at that opening, so 'no collision' is conservative."""
    import mujoco as mj

    m = mj.MjModel.from_xml_path("assets/scene/franka_fr3/fr3_robotiq_robot.xml")
    m.opt.gravity[:] = 0.0
    d = mj.MjData(m)
    home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    d.ctrl[m.actuator("actuator_gripper").id] = angle
    for _ in range(3000):
        d.qpos[:7] = home
        d.qvel[:7] = 0.0
        mj.mj_step(m, d)
    width = float(rq.width_from_angle(d.joint("robotiq_right_driver_joint").qpos[0]))
    tcp = d.site("fr3_gripper_tcp")
    R, p = tcp.xmat.reshape(3, 3), tcp.xpos
    boxes = hand_boxes(width)
    root = m.body("fr3_gripper").id
    for gid in range(m.ngeom):
        b = m.geom_bodyid[gid]
        while b > root:
            b = m.body_parentid[b]
        if b != root or m.geom_group[gid] != 3 or m.geom_type[gid] != mj.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[gid]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        pts = (v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid] - p) @ R  # TCP frame
        inside = np.zeros(len(pts), dtype=bool)
        for ext, ctr in boxes:
            inside |= np.all(np.abs(pts - ctr) <= 0.5 * ext + 0.003, axis=1)
        assert inside.mean() > 0.995, (m.geom(gid).name, width, np.sum(~inside))
