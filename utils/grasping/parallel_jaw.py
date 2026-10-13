"""Parallel-jaw (Franka Hand) grasp geometry and mechanics.

Shared by the grasping tutorials:

* :func:`sample_antipodal_grasps` -- contact pairs on an object mesh that a
  two-finger gripper can squeeze without slipping (friction-cone test).
* :func:`grasp_tcp_poses` -- each contact pair becomes several gripper poses
  (TCP frame), one per approach direction around the closing axis.
* :class:`HandCollisionChecker` -- palm + finger boxes vs the object mesh (FCL).
* :func:`required_grip_force` -- the smallest squeeze force that holds an
  object's weight for a given grasp (soft-finger contact model).

Frames
------
TCP frame of the Franka Hand (URDF ``fr3_hand_tcp``): ``+z`` points from the
hand towards the object (approach), ``+y`` is the finger closing axis, the
origin is midway between the fingertip pads. The hand frame ``fr3_hand`` is the
TCP frame shifted by ``-TCP_OFFSET`` along ``z``.
"""

from __future__ import annotations

from dataclasses import dataclass

import fcl
import numpy as np
import trimesh

# --- Franka Hand geometry (measured from the MuJoCo / URDF model) -----------
MAX_WIDTH = 0.08  # fully open (m)
TCP_OFFSET = 0.1034  # fr3_hand -> fr3_hand_tcp along z (m)
# One finger as a box, in the TCP frame: inner face at y = width/2.
FINGER_EXTENTS = np.array([0.021, 0.0265, 0.054])  # x, y (thickness), z (length)
FINGER_CENTRE_Z = 0.0854 - TCP_OFFSET  # finger reaches 9 mm past the TCP
# Palm housing (same box as the planning model in SimFrankaFR3.FR3_HAND_BOX).
PALM_EXTENTS = np.array([0.065, 0.21, 0.094])
PALM_CENTRE = np.array([0.0, -0.002, 0.02 - TCP_OFFSET])


@dataclass
class AntipodalGrasps:
    """Contact pairs in the object (CAD) frame. Arrays have one row per pair."""

    p1: np.ndarray  # (N, 3) contact points
    p2: np.ndarray
    n1: np.ndarray  # (N, 3) outward surface normals at p1 / p2
    n2: np.ndarray
    width: np.ndarray  # (N,) |p2 - p1|
    cone_angle: np.ndarray  # (N,) worst angle between a normal and the axis

    @property
    def centre(self) -> np.ndarray:
        return 0.5 * (self.p1 + self.p2)

    @property
    def axis(self) -> np.ndarray:
        return (self.p2 - self.p1) / self.width[:, None]

    def __len__(self) -> int:
        return len(self.width)


def sample_antipodal_grasps(
    mesh: trimesh.Trimesh,
    num_samples: int = 3000,
    mu: float = 0.8,
    cone_frac: float = 0.6,
    max_width: float = MAX_WIDTH - 0.01,
    min_width: float = 0.01,
    min_separation: float = 0.008,
    seed: int = 0,
) -> AntipodalGrasps:
    """Sample two-finger contact pairs that pass the friction-cone test.

    For a random surface point ``p1`` with outward normal ``n1``, shoot a ray
    into the object along ``-n1``; where it leaves the surface is ``p2``. The
    pair is *antipodal* if the line ``p1 -> p2`` lies inside both friction
    cones: the angle between the squeeze direction and each inward normal is at
    most ``atan(mu)``. ``cone_frac < 1`` keeps a safety margin inside the cone.
    Pairs whose midpoints are closer than ``min_separation`` (with a similar
    axis) are de-duplicated.
    """
    rng = np.random.default_rng(seed)
    pts, face_idx = trimesh.sample.sample_surface(mesh, num_samples, seed=rng)
    normals = mesh.face_normals[face_idx]

    # Ray from just inside p1 along the inward normal.
    origins = pts - normals * 1e-4
    dirs = -normals
    locs, ray_idx, tri_idx = mesh.ray.intersects_location(
        origins, dirs, multiple_hits=True
    )

    max_angle = cone_frac * np.arctan(mu)
    cos_max = np.cos(max_angle)
    # For each ray keep the farthest exit point within max_width (thick parts
    # are skipped automatically).
    best = {}
    for loc, r, t in zip(locs, ray_idx, tri_idx):
        d = float(np.linalg.norm(loc - pts[r]))
        if d < min_width or d > max_width:
            continue
        if r not in best or d > best[r][0]:
            best[r] = (d, loc, mesh.face_normals[t])

    rows = []
    for r, (d, p2, n2) in best.items():
        p1, n1 = pts[r], normals[r]
        axis = (p2 - p1) / d
        c1 = float(-n1 @ axis)  # finger at p1 pushes along +axis
        c2 = float(n2 @ axis)  # finger at p2 pushes along -axis
        if c1 < cos_max or c2 < cos_max:
            continue
        angle = float(np.arccos(np.clip(min(c1, c2), -1.0, 1.0)))
        rows.append((p1, p2, n1, n2, d, angle))

    # De-duplicate: greedily keep the best-aligned pairs first.
    rows.sort(key=lambda r: r[5])
    kept, kept_rows = [], []
    for row in rows:
        c = 0.5 * (row[0] + row[1])
        a = (row[1] - row[0]) / row[4]
        if any(
            np.linalg.norm(c - kc) < min_separation and abs(a @ ka) > 0.95
            for kc, ka in kept
        ):
            continue
        kept.append((c, a))
        kept_rows.append(row)
    rows = kept_rows

    if not rows:
        empty = np.zeros((0, 3))
        return AntipodalGrasps(empty, empty, empty, empty, np.zeros(0), np.zeros(0))
    p1, p2, n1, n2, w, ang = (np.asarray(x) for x in zip(*rows))
    return AntipodalGrasps(p1, p2, n1, n2, w, ang)


def grasp_tcp_poses(grasps: AntipodalGrasps, num_approach: int = 12):
    """TCP poses (object frame) for every pair and approach direction.

    Returns ``(T, pair_idx)``: ``T`` is ``(N*K, 4, 4)`` with ``y`` along the
    pair axis and ``z`` (approach) swept around it in ``K`` steps; ``pair_idx``
    maps each pose back to its contact pair.
    """
    T_all, idx = [], []
    for i, (c, y) in enumerate(zip(grasps.centre, grasps.axis)):
        ref = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        z0 = np.cross(y, np.cross(ref, y))
        z0 /= np.linalg.norm(z0)
        for k in range(num_approach):
            th = 2.0 * np.pi * k / num_approach
            # Rotate z0 about y by th (Rodrigues; z0 is perpendicular to y).
            z = np.cos(th) * z0 + np.sin(th) * np.cross(y, z0)
            x = np.cross(y, z)
            T = np.eye(4)
            T[:3, :3] = np.column_stack([x, y, z])
            T[:3, 3] = c
            T_all.append(T)
            idx.append(i)
    return np.asarray(T_all), np.asarray(idx, dtype=int)


def hand_boxes(width: float):
    """Palm + two finger boxes as ``[(extents, centre)]`` in the TCP frame,
    with the fingers opened to ``width``."""
    half = 0.5 * width + 0.5 * FINGER_EXTENTS[1]
    return [
        (PALM_EXTENTS, PALM_CENTRE),
        (FINGER_EXTENTS, np.array([0.0, half, FINGER_CENTRE_Z])),
        (FINGER_EXTENTS, np.array([0.0, -half, FINGER_CENTRE_Z])),
    ]


def lowest_hand_point(T_world_tcp: np.ndarray, width: float) -> float:
    """Lowest world z over the palm and finger boxes (for table clearance)."""
    R, p = T_world_tcp[:3, :3], T_world_tcp[:3, 3]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    z_min = np.inf
    for ext, ctr in hand_boxes(width):
        corners = ctr + 0.5 * signs * ext
        z_min = min(z_min, float((corners @ R.T + p)[:, 2].min()))
    return z_min


class HandCollisionChecker:
    """FCL check of the open Franka Hand against an object mesh."""

    def __init__(self, mesh: trimesh.Trimesh):
        bvh = fcl.BVHModel()
        bvh.beginModel(len(mesh.vertices), len(mesh.faces))
        bvh.addSubModel(mesh.vertices, mesh.faces)
        bvh.endModel()
        self._obj = fcl.CollisionObject(bvh, fcl.Transform())

    def collides(self, T_obj_tcp: np.ndarray, width: float, margin: float = 0.0) -> bool:
        """True if palm or fingers (opened to ``width``) come within
        ``margin`` of the object, with the TCP at ``T_obj_tcp`` (object frame)."""
        R, p = T_obj_tcp[:3, :3], T_obj_tcp[:3, 3]
        for ext, ctr in hand_boxes(width):
            box = fcl.CollisionObject(
                fcl.Box(*(ext + 2.0 * margin)), fcl.Transform(R, R @ ctr + p)
            )
            if fcl.collide(box, self._obj, fcl.CollisionRequest(), fcl.CollisionResult()):
                return True
        return False


def required_grip_force(
    T_world_tcp: np.ndarray,
    width: float,
    com_world: np.ndarray,
    mass: float,
    mu: float,
    gamma: float,
    gravity=(0.0, 0.0, -9.81),
    n1_world=None,
    n2_world=None,
    f_max: float = 500.0,
):
    """Smallest squeeze force (N) for which two soft-finger contacts can hold
    the object's weight, given the grasp pose and the object's COM.

    Model (both fingers squeeze with the same force ``F`` along the closing
    axis ``y``; contact 1 is on the ``-y`` side, contact 2 on the ``+y`` side):

    * The weight ``f_g = m g`` acts at the COM. About the TCP the contacts must
      supply the wrench ``(-f_g, -r x f_g)``, ``r = com - p_tcp``.
    * The component along ``y`` is carried by the finger mechanism (the
      fingers are coupled), not by friction.
    * The tangential force ``f_t`` and the torque ``tau_p`` perpendicular to
      ``y`` split over the two contacts, ``width`` apart::

          t_1,2 = f_t / 2  -/+  (tau_p x y) / width

    * The torque ``tau_n`` about ``y`` is shared as torsional friction,
      ``tau_n / 2`` per contact.
    * Contact force ``f_i = +/-F y + t_i``. With inward surface normal ``m_i``
      it must satisfy the elliptic soft-finger limit::

          (|f_i tangential| / mu)^2 + (tau_i / gamma)^2 <= (f_i . m_i)^2

      If the surface is tilted (``m_i`` not along ``y``), part of the squeeze
      itself is tangential, which eats into the friction budget. Without
      normals the surfaces are assumed to face the fingers squarely.

    ``n1_world`` / ``n2_world`` are the *outward* surface normals at the two
    contacts. Returns ``(F_req, details)``; ``F_req = inf`` if no force up to
    ``f_max`` works.
    """
    R, p = T_world_tcp[:3, :3], T_world_tcp[:3, 3]
    y = R[:, 1]
    f_g = mass * np.asarray(gravity, dtype=float)
    r = np.asarray(com_world, dtype=float) - p
    w_f = -f_g
    w_t = -np.cross(r, f_g)

    f_t = w_f - (w_f @ y) * y
    tau_n = float(w_t @ y)
    tau_p = w_t - tau_n * y
    shear = np.cross(tau_p, y) / max(width, 1e-4)
    t = (0.5 * f_t - shear, 0.5 * f_t + shear)
    squeeze_dir = (y, -y)  # finger 1 pushes along +y, finger 2 along -y
    m = (
        y if n1_world is None else -np.asarray(n1_world, dtype=float),
        -y if n2_world is None else -np.asarray(n2_world, dtype=float),
    )
    tors = abs(tau_n) / 2.0

    def margin(F, i):
        f = F * squeeze_dir[i] + t[i]
        mi = m[i] / np.linalg.norm(m[i])
        fn = float(f @ mi)
        if fn <= 0.0:
            return -np.inf
        ft = float(np.linalg.norm(f - fn * mi))
        return fn - float(np.hypot(ft / mu, tors / gamma))

    F_contact = []
    for i in (0, 1):
        if margin(f_max, i) < 0.0:
            F_contact.append(np.inf)
            continue
        lo, hi = 0.0, f_max
        for _ in range(50):  # margin grows with F: bisection
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if margin(mid, i) >= 0.0 else (mid, hi)
        F_contact.append(hi)

    details = {
        "f_tangential": float(np.linalg.norm(f_t)),
        "f_axial": float(abs(w_f @ y)),
        "tau_axis": abs(tau_n),
        "tau_perp": float(np.linalg.norm(tau_p)),
        "F_contact": tuple(F_contact),
    }
    return max(F_contact), details
