"""Graspable objects: load the mesh / physics from configs/objects/*.yaml and
add the object to a MuJoCo scene."""

from __future__ import annotations

import mujoco as mj
import numpy as np
import trimesh
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation


class GraspObject:
    """An object description shared by the env, the controllers and the
    grasp-quality script, so they all agree on mesh, mass and frame.

    The object (CAD) frame is the mesh frame. ``spawn_rpy`` rotates it so the
    object stands upright on the table.
    """

    def __init__(self, cfg):
        if isinstance(cfg, str):
            cfg = OmegaConf.load(cfg)
        self.cfg: DictConfig = cfg
        self.name = str(cfg["name"])
        self.mesh_path = str(cfg["mesh_path"])
        self.mass = float(cfg["mass"])
        self.rgba = list(cfg.get("rgba", [0.6, 0.6, 0.6, 1.0]))
        self.friction = list(cfg.get("friction", [0.5, 0.005, 0.0001]))
        self.R_spawn = Rotation.from_euler(
            "xyz", list(cfg.get("spawn_rpy", [0.0, 0.0, 0.0]))
        ).as_matrix()

        self.mesh: trimesh.Trimesh = trimesh.load(self.mesh_path, force="mesh")
        # Grasp sampling needs a closed surface: rays shot into an open mesh
        # can exit through internal faces. Fall back to the convex hull, which
        # is exact for cans / boxes and conservative otherwise.
        self.grasp_mesh = (
            self.mesh if self.mesh.is_watertight else self.mesh.convex_hull
        )

    def com_cad(self) -> np.ndarray:
        """Centre of mass in the object frame (uniform density, hull volume)."""
        return np.asarray(self.mesh.convex_hull.center_mass, dtype=float)

    def resting_height(self, R_world_obj: np.ndarray, floor_z: float = 0.0) -> float:
        """Object-frame origin height at which the object rests on the floor."""
        z = (self.mesh.vertices @ R_world_obj.T)[:, 2]
        return floor_z - float(z.min())

    def add_to_spec(self, spec: mj.MjSpec, body_name: str = "grasp_object", pos=(0.5, 0.0, 0.1)):
        """Add a free-floating body with this object's mesh to ``spec``.

        The body frame equals the object (CAD) frame. MuJoCo collides with the
        mesh's convex hull; the mass is set explicitly (YCB value).
        """
        mesh_name = f"{body_name}_mesh"
        spec.add_mesh(name=mesh_name, file=self._abs(self.mesh_path))
        body = spec.worldbody.add_body(name=body_name, pos=list(pos))
        body.add_freejoint(name=f"{body_name}_free")
        geom = body.add_geom(
            name=f"{body_name}_geom",
            type=mj.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            rgba=self.rgba,
            friction=self.friction,
        )
        geom.mass = self.mass
        return body

    @staticmethod
    def _abs(path: str) -> str:
        import os

        return os.path.abspath(path)
