"""MuJoCo grasping env: robot + collision info + one free object on the table."""

from __future__ import annotations

import copy

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from data_type.basic_types.NamedVecListData import NamedVecListData
from data_type.basic_types.SE3PoseData import SE3PoseData
from env.MujocoCamEnv import CameraMixin
from env.MujocoColEnv import MujocoColEnv
from utils.grasping.objects import GraspObject


class MujocoGraspEnv(MujocoColEnv):
    """Collision env plus a free ``grasp_object`` built from an object config
    (``configs/objects/*.yaml``).

    Published (extr):

    * ``object_pose_channel``: world SE(3) of the object frame (= mesh frame).
      On the real robot this comes from perception (tutorial 13).
    * ``object_physics_channel``: ``[mass, com_obj(3), gravity_world(3), pad(5)]``.

    Keys: ``o`` re-place the object at a random XY / yaw (no reset);
    ``r`` full reset.
    """

    def __init__(self, config, *args, **kwargs):
        # Set before super().__init__: the parent calls reset() / _modify_mjspec().
        self.obj = GraspObject(config["object_config"])
        self._object_pose_channel = config["pub_manager"].get("object_pose_channel", "")
        self._object_physics_channel = config["pub_manager"].get(
            "object_physics_channel", ""
        )
        self._object_pose_dt = float(config.get("object_pose_update_dt", 1.0 / 30.0))
        self._last_object_pose_t = -1e9
        self._object_pose_data = SE3PoseData(name=self._object_pose_channel)
        self._object_physics_data = NamedVecListData(
            num_vecs=1, vec_dim=12, name=self._object_physics_channel
        )
        self._spawn_xy_center = np.asarray(config.get("obj_spawn_xy_center", [0.5, 0.0]))
        self._spawn_xy_half = np.asarray(config.get("obj_spawn_xy_half", [0.05, 0.08]))
        self._spawn_yaw_half = float(config.get("obj_spawn_yaw_half", np.pi))
        self._rng = np.random.default_rng(int(config.get("obj_spawn_seed", 0)) or None)
        self._object_body_id = -1

        super().__init__(config, *args, **kwargs)
        print(
            f"[GraspEnv] object={self.obj.name} ({self.obj.mass} kg). "
            "Key 'o' = re-place object; 'r' = full reset."
        )

    # ------------------------------------------------------------------ scene
    def _modify_mjspec(self) -> None:
        self.obj.add_to_spec(self.mj_spec)
        opt = self.mj_spec.option
        # Grasping-friendly contact settings (see run_grasp_quality.py):
        # elliptic friction cones and the noslip solver, so a held object
        # does not creep out of the fingers.
        opt.cone = mj.mjtCone.mjCONE_ELLIPTIC
        opt.impratio = 10.0
        opt.noslip_iterations = 5

    def reset(self) -> None:
        super().reset()
        self._object_body_id = self.mj_model.body("grasp_object").id
        self._place_object()

    def _handle_key(self, key: str) -> None:
        if key in ("o", "O"):
            self._place_object()
            return
        super()._handle_key(key)

    def _key_callback(self, keycode) -> None:
        if keycode in (ord("o"), ord("O")):
            self._place_object()
            return
        super()._key_callback(keycode)

    def _place_object(self) -> None:
        """Random XY + yaw, standing upright on the table, at rest."""
        jid = self.mj_model.body_jntadr[self._object_body_id]
        qadr = int(self.mj_model.jnt_qposadr[jid])
        dadr = int(self.mj_model.jnt_dofadr[jid])

        xy = self._spawn_xy_center + self._rng.uniform(
            -self._spawn_xy_half, self._spawn_xy_half
        )
        yaw = float(self._rng.uniform(-self._spawn_yaw_half, self._spawn_yaw_half))
        R = Rotation.from_euler("z", yaw).as_matrix() @ self.obj.R_spawn
        z = self.obj.resting_height(R) + 0.001
        q = Rotation.from_matrix(R).as_quat()  # xyzw

        def _apply():
            self.mj_data.qpos[qadr : qadr + 3] = [xy[0], xy[1], z]
            self.mj_data.qpos[qadr + 3 : qadr + 7] = [q[3], q[0], q[1], q[2]]
            self.mj_data.qvel[dadr : dadr + 6] = 0.0
            mj.mj_forward(self.mj_model, self.mj_data)

        viewer = getattr(self, "mj_viewer", None)
        if viewer is not None and viewer.is_running():
            with viewer.lock():
                _apply()
        else:
            _apply()
        self._last_object_pose_t = -1e9  # publish the new pose right away
        print(f"[GraspEnv] object at xy=({xy[0]:.3f}, {xy[1]:.3f}), yaw={yaw:.2f} rad")

    # -------------------------------------------------------------- publishing
    def _object_T(self) -> np.ndarray:
        b = self._object_body_id
        T = np.eye(4)
        T[:3, :3] = self.mj_data.xmat[b].reshape(3, 3)
        T[:3, 3] = self.mj_data.xpos[b]
        return T

    def _object_physics(self) -> np.ndarray:
        b = self._object_body_id
        v = np.zeros(12)
        v[0] = float(self.mj_model.body_mass[b])
        v[1:4] = self.mj_model.body_ipos[b]
        v[4:7] = self.mj_model.opt.gravity
        return v

    @staticmethod
    def _put_latest(que, item) -> None:
        while not que.empty():
            try:
                que.get_nowait()
            except Exception:
                break
        que.put(item)

    def _sync_data_from_sim(self, intr_pub_que_dict, extr_pub_que_dict, time_pub_que):
        super()._sync_data_from_sim(intr_pub_que_dict, extr_pub_que_dict, time_pub_que)
        t = float(self.mj_data.time)
        if t - self._last_object_pose_t < self._object_pose_dt:
            return
        self._last_object_pose_t = t
        if self._object_pose_channel:
            self._object_pose_data.set_from_matrix(t, self._object_T())
            self._put_latest(
                extr_pub_que_dict[self._object_pose_channel],
                copy.deepcopy(self._object_pose_data),
            )
        if self._object_physics_channel:
            self._object_physics_data.set_data(
                t, ["physics_params"], self._object_physics().reshape(1, -1)
            )
            self._put_latest(
                extr_pub_que_dict[self._object_physics_channel],
                copy.deepcopy(self._object_physics_data),
            )


class MujocoGraspCamEnv(CameraMixin, MujocoGraspEnv):
    """Grasping env plus fixed RGB-D cameras: rehearse the real-robot pipeline
    camera -> FoundationPose -> run_pose_bridge.py -> grasp controller in sim.
    Set ``pub_manager.object_pose_channel: ""`` so the ground-truth pose is not
    published on the channel the bridge uses."""
