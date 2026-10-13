"""Planning collision model (capsules + gripper boxes) used by every
collision-aware controller, and the collision pairs in the configs."""

import glob

import mujoco as mj
import numpy as np
import pytest
from omegaconf import OmegaConf

from robot_platform.sim.franka_fr3.SimFrankaFR3 import (
    ARM_JOINT_NAMES,
    FR3_COL_CAPSULES,
    FR3_GRIPPER_BOXES,
    fr3_col_primitives,
)
from utils.collision.wall_geoms import extract_static_col_geoms
from utils.fcl.getter import col_frame_name

SCENE = "assets/scene/franka_fr3/scene_default.xml"
LINKS = [f"fr3_link{i}" for i in range(8)]


def _world(data, body):
    T = np.eye(4)
    b = data.body(body)
    T[:3, :3], T[:3, 3] = b.xmat.reshape(3, 3), b.xpos
    return T


def _inside(points, prims, data, tol):
    """Boolean mask: point inside (+tol) any primitive."""
    ok = np.zeros(len(points), dtype=bool)
    for p in prims:
        T = _world(data, col_frame_name(p["name"])) @ p["offset"]
        local = (points - T[:3, 3]) @ T[:3, :3]
        if p["geom_type"] == "capsule":
            r, length = p["size"]
            z = np.clip(local[:, 2], -length / 2, length / 2)
            d = np.linalg.norm(local - np.c_[np.zeros((len(z), 2)), z], axis=1)
            ok |= d <= r + tol
        else:
            ok |= np.all(np.abs(local) <= np.asarray(p["size"]) / 2 + tol, axis=1)
    return ok


def test_primitive_names_map_to_links():
    names = [p["name"] for p in fr3_col_primitives()]
    assert len(names) == len(set(names)) == len(FR3_COL_CAPSULES) + len(FR3_GRIPPER_BOXES)
    model = mj.MjModel.from_xml_path(SCENE)
    for n in names:
        assert model.body(col_frame_name(n)) is not None


@pytest.mark.parametrize("seed", range(3))
def test_capsules_cover_arm_collision_meshes(seed):
    """Every vertex of the arm links' collision meshes lies inside the planning
    primitives (at random poses), so the FCL distances are conservative."""
    model = mj.MjModel.from_xml_path(SCENE)
    data = mj.MjData(model)
    rng = np.random.default_rng(seed)
    for n in ARM_JOINT_NAMES:
        j = model.joint(n)
        data.qpos[j.qposadr[0]] = rng.uniform(*j.range)
    mj.mj_kinematics(model, data)

    prims = fr3_col_primitives()
    for link in LINKS[1:]:  # link0 is fixed to the base / table
        gid = model.geom(f"{link}_collision").id
        mid = model.geom_dataid[gid]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[a:a + n]
        R = data.geom_xmat[gid].reshape(3, 3)
        pts = verts @ R.T + data.geom_xpos[gid]
        inside = _inside(pts, prims, data, tol=0.005)
        assert inside.mean() > 0.999, f"{link}: {np.sum(~inside)} vertices outside"


def test_gripper_boxes_cover_gripper_collision_meshes():
    """The gripper's collision meshes (fully open: the widest it gets) lie
    inside its two planning boxes."""
    model = mj.MjModel.from_xml_path(SCENE)
    data = mj.MjData(model)
    for n, q in zip(ARM_JOINT_NAMES, [0, -0.785, 0, -2.356, 0, 1.571, 0.785]):
        data.qpos[model.joint(n).qposadr[0]] = q
    mj.mj_kinematics(model, data)
    prims = [p for p in fr3_col_primitives() if p["name"].startswith("fr3_gripper")]
    root = model.body("fr3_gripper").id
    for gid in range(model.ngeom):
        b = model.geom_bodyid[gid]
        while b > root:
            b = model.body_parentid[b]
        if b != root or model.geom_group[gid] != 3 or model.geom_type[gid] != mj.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        pts = model.mesh_vert[a:a + n] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        inside = _inside(pts, prims, data, tol=0.002)
        assert inside.all(), f"{model.geom(gid).name}: {np.sum(~inside)} vertices outside"


def _config_pairs():
    files = glob.glob("configs/franka_fr3/controllers/*.yaml") + glob.glob(
        "configs/visualizer/franka_fr3/*.yaml"
    )
    out = []
    for f in files:
        cfg = OmegaConf.to_container(OmegaConf.load(f))
        stack = [cfg]
        while stack:
            c = stack.pop()
            if isinstance(c, dict):
                for k, v in c.items():
                    if k in ("col_pairs", "self_col_pairs") and v:
                        out += [(f, k, tuple(p)) for p in v]
                    else:
                        stack.append(v)
    return out


def test_config_col_pairs_refer_to_known_geoms():
    robot = {p["name"] for p in fr3_col_primitives()}
    static = set()
    for scene in glob.glob("assets/scene/franka_fr3/scene_*.xml"):
        static |= {g["name"] for g in extract_static_col_geoms(mj.MjModel.from_xml_path(scene))}
    pairs = _config_pairs()
    assert pairs
    for f, key, (a, b) in pairs:
        assert a in robot, (f, a)
        if key == "self_col_pairs":
            assert b in robot, (f, b)
            assert col_frame_name(a) != col_frame_name(b)
        else:
            # grasping configs also name the object, published at runtime
            assert b in static or b.startswith("grasp_object") or b == "object", (f, b)
