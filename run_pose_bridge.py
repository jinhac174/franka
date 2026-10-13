"""Bridge perception -> grasping: object pose from FoundationPose to the
grasp controllers' ``franka_object_pose`` (and ``franka_object_physics``).

FoundationPose (``perception/foundation_pose``) publishes, as a ``vec_list_t``
``[x, y, z, R (row-major 3x3)]``, the pose of the object's *oriented bounding
box* frame in the camera's "world" frame (``inv(extrinsic)`` of the camera
info). The grasp controllers want the pose of the object's *mesh* frame in the
*robot base* frame, as ``se3_pose_t``. This node converts:

    T_base_mesh = T_base_world @ T_world_bb @ to_origin

* ``to_origin`` = ``trimesh.bounds.oriented_bounds(mesh)``, from the SAME mesh
  file FoundationPose loads (``fp_mesh_file``).
* ``T_base_world`` = identity when the camera extrinsics are calibrated against
  the robot base (tutorial 14); otherwise set ``base_from_world``.

It also republishes the object's mass / COM from its object config, like the
simulator's ``franka_object_physics`` (used for logging; tutorial 12 weighs).

    python run_pose_bridge.py --config configs/franka_fr3/perception/pose_bridge.yaml
"""

import argparse
import time

import lcm
import numpy as np
import trimesh
from omegaconf import OmegaConf

from lcm_type.pose.se3_pose_t import se3_pose_t
from lcm_type.vector.vec_list_t import vec_list_t
from scipy.spatial.transform import Rotation
from utils.grasping.objects import GraspObject
from utils.terminal.argparse_utils import parse_nested_args, parse_unknown_args


def bb_pose_to_mesh_pose(vec12, to_origin, T_base_world=np.eye(4)):
    """``[p, R row-major]`` of the bounding-box frame -> 4x4 mesh-frame pose."""
    T_world_bb = np.eye(4)
    T_world_bb[:3, 3] = vec12[:3]
    T_world_bb[:3, :3] = np.asarray(vec12[3:12]).reshape(3, 3)
    return T_base_world @ T_world_bb @ to_origin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), parse_nested_args(parse_unknown_args(unknown)))
    print(OmegaConf.to_yaml(cfg))

    fp_mesh = trimesh.load(cfg["fp_mesh_file"], force="mesh")
    to_origin, _ = trimesh.bounds.oriented_bounds(fp_mesh)
    T_base_world = np.asarray(cfg.get("base_from_world", np.eye(4).tolist()), dtype=float)
    obj = GraspObject(cfg["object_config"])
    physics = np.zeros(12)
    physics[0], physics[1:4], physics[4:7] = obj.mass, obj.com_cad(), (0.0, 0.0, -9.81)

    lc = lcm.LCM()
    out_pose, out_phys = cfg["object_pose_channel"], cfg.get("object_physics_channel", "")
    state = {"n": 0, "last_phys": 0.0}

    def on_pose(channel, data):
        msg = vec_list_t.decode(data)
        if msg.num_vecs < 1:
            return
        vec = np.asarray(msg.vec_list[0], dtype=float)
        T = bb_pose_to_mesh_pose(vec, to_origin, T_base_world)
        out = se3_pose_t()
        out.timestamp = msg.timestamp
        out.position = T[:3, 3].tolist()
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        out.quat_wxyz = [w, x, y, z]
        lc.publish(out_pose, out.encode())
        state["n"] += 1
        if out_phys and time.time() - state["last_phys"] > 0.5:
            p = vec_list_t()
            p.timestamp, p.num_vecs, p.vec_dim = msg.timestamp, 1, 12
            p.name_list = ["physics_params"]
            p.vec_list = [physics.tolist()]
            lc.publish(out_phys, p.encode())
            state["last_phys"] = time.time()

    lc.subscribe(cfg["fp_pose_channel"], on_pose)
    print(f"[PoseBridge] {cfg['fp_pose_channel']} (bounding box, camera world) -> "
          f"{out_pose} (mesh frame, robot base)")
    t_report = time.time()
    try:
        while True:
            lc.handle_timeout(100)
            if time.time() - t_report > 5.0:
                print(f"[PoseBridge] {state['n'] / (time.time() - t_report):.1f} poses/s")
                state["n"], t_report = 0, time.time()
    except KeyboardInterrupt:
        print("[PoseBridge] stopped.")


if __name__ == "__main__":
    main()
