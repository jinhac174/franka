"""Reactive parallel-jaw grasping for the FR3 + Robotiq 2F-85 (tutorial 10).

Pipeline
--------
1. Offline (startup): sample antipodal contact pairs on the object mesh, turn
   each into gripper TCP poses (several approach directions, both finger
   orders), drop the ones where the open hand would hit the object, and
   predict the grip force each needs (``utils/grasping/parallel_jaw.py``).
2. Online (plan thread): place the candidates with the live object pose, filter
   by table clearance / approach direction, score them, and keep the selection
   *sticky* so the target does not jump while the object moves.
3. Reach: solve IK for the pre-grasp pose (backed off along the approach
   axis, tutorial 09) and drive the joints there (tutorial 07 style), through
   the collision-aware QP of tutorial 06. An unreachable grasp is caught by the
   IK right away and the next candidate is tried.
4. Approach: a 6-D pose velocity field moves the TCP straight in from the
   pre-grasp, again through the collision-aware QP.
5. ``grasp(width, speed, force)`` on the gripper, then lift.

Keyboard (controller terminal)
------------------------------
``v``  start / restart the grasp     ``x``  release (open) and stop
``h``  joint-space home               ``w/s a/d r/f``  nudge TCP (as tutorial 06)
``p`` / ``q``  pause / quit
"""

from __future__ import annotations

import threading
import time

import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from controller.IKQPControl import IKQPController, PlanResult
from data_type.basic_types.GripperCmdData import GripperCmdData
from data_type.basic_types.GripperStateData import GripperStateData
from data_type.basic_types.JointCtrlData import JointCtrlData
from data_type.basic_types.NamedVecListData import NamedVecListData
from data_type.basic_types.SE3PoseData import SE3PoseData
from utils.grasping.objects import GraspObject
from utils.grasping.parallel_jaw import (
    MAX_WIDTH,
    HandCollisionChecker,
    grasp_tcp_poses,
    hand_boxes,
    required_grip_force,
    sample_antipodal_grasps,
)
from utils.pinocchio.getter import get_fk_link_pose, get_link_Jacobian
from utils.pinocchio.ik_se3 import solve_frame_ik_se3

_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
# 180 deg about the TCP z axis: swaps the fingers, same grasp.
_FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])


def _translate(T: np.ndarray, d_local) -> np.ndarray:
    out = T.copy()
    out[:3, 3] = T[:3, 3] + T[:3, :3] @ np.asarray(d_local, dtype=float)
    return out


def _sat(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v * (vmax / n) if n > vmax else v


class ParallelJawGraspController(IKQPController):
    """Antipodal grasp selection + pose IK QP reaching + gripper grasp + lift."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        g = config["grasp"]
        self._grasp_cfg = g

        self._obj = GraspObject(config["object_config"])
        self._obj_mass = self._obj.mass  # replaced by the env's value if published

        sm, pm = config["sub_manager"], config["pub_manager"]
        self._gripper_state_channel = sm["gripper_state_channel"]
        self._object_pose_channel = sm["object_pose_channel"]
        self._object_physics_channel = sm.get("object_physics_channel", "")
        self._gripper_cmd_channel = pm["gripper_cmd_channel"]
        self._target_tcp_channel = pm.get("target_tcp_channel", "")
        self._candidates_channel = pm.get("grasp_candidates_channel", "")

        # Grasp model / candidate generation
        self._mu = float(g.get("mu", 0.8))
        self._gamma = float(g.get("gamma", 0.008))
        self._open_margin = float(g.get("open_margin", 0.02))
        self._min_downward = float(g.get("min_downward", 0.3))
        self._table_z = float(g.get("table_z", 0.0))
        self._table_clearance = float(g.get("table_clearance", 0.02))
        self._max_reach = float(g.get("max_reach", 0.75))
        # Scoring (lower is better) and stickiness
        self._w_force = float(g.get("w_force", 1.0))
        self._f_ref = float(g.get("f_ref", 10.0))
        self._w_rot = float(g.get("w_rot", 1.0))
        self._w_pos = float(g.get("w_pos", 2.0))
        self._switch_margin = float(g.get("switch_margin", 0.3))
        self._switch_ticks = int(g.get("switch_ticks", 10))
        # Motion
        self._pregrasp_dist = float(g.get("pregrasp_dist", 0.10))
        self._lift_height = float(g.get("lift_height", 0.15))
        self._pos_gain = float(g.get("pos_gain", 3.0))
        self._rot_gain = float(g.get("rot_gain", 2.0))
        self._v_reach = float(g.get("v_reach", 0.15))
        self._v_approach = float(g.get("v_approach", 0.05))
        self._w_max = float(g.get("w_max", 0.8))
        self._rot_weight = float(g.get("rot_weight", 0.3))
        self._tol_pre = (float(g.get("pregrasp_pos_tol", 0.01)), float(g.get("pregrasp_rot_tol", 0.05)))
        self._tol_grasp = (float(g.get("grasp_pos_tol", 0.004)), float(g.get("grasp_rot_tol", 0.03)))
        self._object_moved_tol = float(g.get("object_moved_tol", 0.02))
        # Gripper
        self._grasp_force = float(g.get("grasp_force", 30.0))
        self._grasp_speed = float(g.get("grasp_speed", 0.1))
        self._grasp_eps = float(g.get("grasp_eps", 0.01))
        self._gripper_timeout = float(g.get("gripper_timeout", 3.0))
        self._approach_timeout = float(g.get("approach_timeout", 8.0))
        # Reach = IK to the pre-grasp + joint-space motion.
        self._reach_joint_gain = float(g.get("reach_joint_gain", 2.0))
        self._reach_joint_tol = float(g.get("reach_joint_tol", 0.01))
        self._ik_limit_margin = float(g.get("ik_limit_margin", 0.05))
        # The pre-grasp only needs ~mm accuracy (the approach corrects the rest).
        self._ik_tol = float(g.get("ik_tol", 1e-3))
        self._ik_max_iters = int(g.get("ik_max_iters", 500))
        self._ik_min_sigma = float(g.get("ik_min_sigma", 0.08))
        self._ik_frame = self._pin_model.getFrameId(self._task_link)
        self._q_pre = None
        self._ik_for = None  # (selection, pre-grasp position) the IK was solved for
        # Progress watchdog: no 5 mm improvement within stall_time -> give up on
        # this goal (typically a joint limit the scoring could not foresee).
        self._stall_time = float(g.get("stall_time", 3.0))
        self._stall_dist = float(g.get("stall_dist", 0.005))
        self._stall_angle = float(g.get("stall_angle", 0.05))
        self._progress = (np.inf, np.inf, time.time())
        self._last_goal = None
        self._actual_pos_tol = float(g.get("actual_pos_tol", 0.008))

        # Reference tracking for the arm (see _update)
        self._hold_kp = np.asarray(config["hold_kp"], dtype=np.float64)
        self._hold_kd = np.asarray(config["hold_kd"], dtype=np.float64)
        self._max_ref_err = float(config.get("max_ref_err", 0.05))
        self._q_ref = None  # control thread
        self._q_ref_shared = None  # copy for the planner (under _meas_lock)
        # Stale-plan guard: if the planner falls behind (CPU load), stop
        # integrating its last joint velocity instead of coasting on it.
        self._max_plan_age = float(config.get("max_plan_age", 0.05))
        self._plan_stamp = 0.0

        # Latest measurements from the 1 kHz loop (guarded by _meas_lock).
        self._T_obj = None
        self._gripper_state: GripperStateData | None = None
        # State machine (guarded by _fsm_lock: plan thread + keyboard only, so
        # candidate scoring never blocks the control loop).
        self._fsm_lock = threading.Lock()
        self._state = "idle"
        self._sel = None
        self._switch_count = 0
        self._failed: set[int] = set()
        self._gripper_seq = 0
        self._wait_seq = None
        self._wait_t0 = 0.0
        self._T_goal = None
        self._T_obj_at_lock = None
        self._T_lift = None
        self._T_backoff = None
        self._T_viz = None
        self._q_plan = None

        self._build_candidates()
        print(
            "[PJGrasp] Keys: v=grasp, x=release, h=home, w/s/a/d/r/f nudge, p/q. "
            f"{len(self._cand_T)} candidate TCP poses for {self._obj.name}."
        )

    # ------------------------------------------------------------ candidates
    def _build_candidates(self) -> None:
        """Antipodal pairs -> collision-free TCP poses (object frame) + the grip
        force each one needs to hold the object upright (gravity along the
        object's spawn 'down', which a yaw rotation does not change)."""
        cfg = self._cfg_grasp_sampling()
        pairs = sample_antipodal_grasps(
            self._obj.grasp_mesh, num_samples=cfg["num_samples"], mu=self._mu, seed=cfg["seed"]
        )
        T_all, idx = grasp_tcp_poses(pairs, cfg["num_approach"])
        # Check the open hand against the object's convex hull: that is what
        # MuJoCo collides with, and on a real object it is conservative.
        chk = HandCollisionChecker(self._obj.mesh.convex_hull)
        g_obj = self._obj.R_spawn.T @ np.array([0.0, 0.0, -9.81])
        com = self._obj.com_cad()

        T_c, width, F_req, p1, p2 = [], [], [], [], []
        # The open hand must be clear at the grasp pose *and* along the straight
        # approach from the pre-grasp: on a tilted approach the fingers can
        # clip the object on the way in even if the final pose is free.
        offsets = np.linspace(0.0, self._pregrasp_dist, 6)
        for T, i in zip(T_all, idx):
            w = float(pairs.width[i])
            ow = min(w + self._open_margin, MAX_WIDTH)
            if any(chk.collides(_translate(T, [0, 0, -d]), ow, margin=0.003) for d in offsets):
                continue
            F, _ = required_grip_force(
                T, w, com, self._obj_mass, self._mu, self._gamma,
                gravity=g_obj, n1_world=pairs.n1[i], n2_world=pairs.n2[i],
            )
            for Tk in (T, T @ _FLIP):  # both finger orders
                T_c.append(Tk)
                width.append(w)
                F_req.append(F)
                p1.append(pairs.p1[i])
                p2.append(pairs.p2[i])
        self._cand_T = np.asarray(T_c)
        self._cand_width = np.asarray(width)
        self._cand_F = np.asarray(F_req)
        self._cand_p = (np.asarray(p1), np.asarray(p2))
        # Hand box corners (TCP frame) at each candidate's opening width.
        corners = []
        for w in self._cand_width:
            ow = min(w + self._open_margin, MAX_WIDTH)
            corners.append(np.vstack([c + 0.5 * _SIGNS * e for e, c in hand_boxes(ow)]))
        self._cand_corners = np.asarray(corners)  # (N, 24, 3)

    def _cfg_grasp_sampling(self) -> dict:
        g = self._grasp_cfg
        return {
            "num_samples": int(g.get("num_samples", 3000)),
            "num_approach": int(g.get("num_approach", 12)),
            "seed": int(g.get("seed", 0)),
        }

    def _score_candidates(self, T_obj: np.ndarray, T_tcp: np.ndarray):
        """World poses, validity mask and score (lower is better)."""
        T_w = np.einsum("ij,njk->nik", T_obj, self._cand_T)
        R, p = T_w[:, :3, :3], T_w[:, :3, 3]
        down = -R[:, 2, 2]  # approach axis . (0, 0, -1)
        z_low = (np.einsum("nij,nkj->nki", R, self._cand_corners) + p[:, None, :])[..., 2].min(1)
        pre_z = p[:, 2] - self._pregrasp_dist * R[:, 2, 2]
        valid = (
            (down >= self._min_downward)
            & (z_low >= self._table_z + self._table_clearance)
            & (np.linalg.norm(p[:, :2], axis=1) <= self._max_reach)
            & (pre_z >= self._table_z + 0.05)
        )
        if self._failed:
            valid[list(self._failed)] = False
        R_rel = np.einsum("ji,njk->nik", T_tcp[:3, :3], R)
        rot = np.arccos(np.clip((np.trace(R_rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
        pos = np.linalg.norm(p - T_tcp[:3, 3], axis=1)
        score = (
            self._w_force * self._cand_F / self._f_ref
            + self._w_rot * rot
            + self._w_pos * pos
        )
        return T_w, valid, score

    def _select(self, T_obj: np.ndarray, T_tcp: np.ndarray):
        """Sticky selection: switch only if another grasp is clearly better
        for ``switch_ticks`` consecutive plans, or the current one is invalid.
        Once the pre-grasp IK is solved for a grasp, stay with it."""
        T_w, valid, score = self._score_candidates(T_obj, T_tcp)
        if not valid.any():
            return None, T_w
        cur = self._sel
        # Committed: IK already solved for this grasp (reach under way). Keep it
        # while it stays valid; switching now would restart the reach.
        if cur is not None and valid[cur] and self._ik_for is not None and self._ik_for[0] == cur:
            return cur, T_w
        best = int(np.argmin(np.where(valid, score, np.inf)))
        if cur is None or not valid[cur]:
            self._sel, self._switch_count = best, 0
        elif score[best] < score[cur] - self._switch_margin:
            self._switch_count += 1
            if self._switch_count >= self._switch_ticks:
                self._sel, self._switch_count = best, 0
        else:
            self._switch_count = 0
        return self._sel, T_w

    # ------------------------------------------------------------ data in
    def _check_and_get_data_from_que(self) -> None:
        super()._check_and_get_data_from_que()
        gs = self._latest(self.intr_sub_que_dict, self._gripper_state_channel)
        pose = self._latest(self.extr_sub_que_dict, self._object_pose_channel)
        phys = self._latest(self.extr_sub_que_dict, self._object_physics_channel)
        with self._meas_lock:
            if isinstance(gs, GripperStateData):
                self._gripper_state = gs
            if isinstance(pose, SE3PoseData):
                _, pos, wxyz = pose.get_data()
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat(np.roll(wxyz, -1)).as_matrix()
                T[:3, 3] = pos
                self._T_obj = T
        if isinstance(phys, NamedVecListData):
            m = float(phys.get_data()[2][0][0])
            if m > 0 and abs(m - self._obj_mass) > 1e-6:
                print(f"[PJGrasp] Object mass from env: {m:.3f} kg")
                self._obj_mass = m

    @staticmethod
    def _latest(que_dict, channel):
        if not channel or channel not in que_dict:
            return None
        que, item = que_dict[channel], None
        while not que.empty():
            item = que.get()
        return item

    # ------------------------------------------------------------ gripper out
    def _gripper(self, mode: int, width: float = 0.0, force: float = 0.0) -> int:
        self._gripper_seq += 1
        cmd = GripperCmdData()
        cmd.set_data(
            self._cur_time, self._gripper_seq, mode, width, self._grasp_speed,
            force, self._grasp_eps, self._grasp_eps,
        )
        self.pub_que_dict[self._gripper_cmd_channel].put(cmd)
        return self._gripper_seq

    def _gripper_done(self):
        """``None`` while the last command runs, else its success flag."""
        gs = self._gripper_state
        if gs is None or self._wait_seq is None:
            return None
        if gs.last_seq == self._wait_seq and not gs.busy:
            return bool(gs.last_success)
        if time.time() - self._wait_t0 > self._gripper_timeout:
            print("[PJGrasp] Gripper command timed out.")
            return False
        return None

    def _wait_for(self, seq: int) -> None:
        self._wait_seq, self._wait_t0 = seq, time.time()

    # ------------------------------------------------------------ keyboard
    def _handle_key(self, key: str) -> None:
        if key == "v":
            with self._meas_lock:
                have_obj = self._T_obj is not None
            if not have_obj:
                print("[PJGrasp] No object pose yet.")
                return
            with self._fsm_lock:
                self._failed.clear()
                self._sel = None
                self._enter("reach")
            self._wait_for(self._gripper(GripperCmdData.MOVE, MAX_WIDTH))
            return
        if key == "x":
            self._gripper(GripperCmdData.MOVE, MAX_WIDTH)
            with self._fsm_lock:
                self._enter("idle")
            with self._meas_lock:
                self._target_initialized = False
            return
        if key in ("h", "w", "s", "a", "d", "r", "f", "[", "]"):
            with self._fsm_lock:
                self._enter("idle")
        super()._handle_key(key)

    def _enter(self, state: str) -> None:
        if state != self._state:
            print(f"[PJGrasp] {self._state} -> {state}")
            self._progress = (np.inf, np.inf, time.time())
        self._state = state

    def _stalled(self, T_a, T_b) -> bool:
        """True if neither the position nor the orientation error has improved
        (by ``stall_dist`` m / ``stall_angle`` rad) for ``stall_time`` s.

        Either counts as progress: during a large reorientation the QP may
        trade some position error for rotation, and vice versa. A jump of the
        goal itself (new grasp, object moved) restarts the clock.
        """
        if self._last_goal is not None and np.linalg.norm(T_b[:3, 3] - self._last_goal) > 0.02:
            self._progress = (np.inf, np.inf, time.time())
        self._last_goal = T_b[:3, 3].copy()
        dp = float(np.linalg.norm(T_b[:3, 3] - T_a[:3, 3]))
        dR = T_a[:3, :3].T @ T_b[:3, :3]
        ang = float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
        best_p, best_a, t_best = self._progress
        improved = False
        if dp < best_p - self._stall_dist:
            best_p, improved = dp, True
        if ang < best_a - self._stall_angle:
            best_a, improved = ang, True
        if improved:
            self._progress = (min(best_p, dp), min(best_a, ang), time.time())
            return False
        return time.time() - t_best > self._stall_time

    # ------------------------------------------------------------ planning
    def _compute_plan(self) -> PlanResult:
        with self._meas_lock:
            T_obj = None if self._T_obj is None else self._T_obj.copy()
        with self._fsm_lock:
            state = self._state
        if state == "idle":
            return super()._compute_plan()

        q, qd, _, _, _ = self._snapshot_for_plan()
        T_tcp = get_fk_link_pose(q, self._pin_model, self._pin_data_plan, self._task_link)
        # Measured TCP (q_ref above is the reference the arm is tracking).
        q_meas = IKQPController._snapshot_for_plan(self)[0]
        self._T_tcp_meas = get_fk_link_pose(
            q_meas, self._pin_model, self._pin_data_plan, self._task_link
        )
        hold = PlanResult(
            qd_des=np.zeros(self._n_x), tip=T_tcp[:3, 3], velocity=np.zeros(3),
            mode=state, at_goal=True, success=True,
        )

        with self._fsm_lock:
            self._q_plan = q
            self._T_viz = None
            T_goal, v_max = self._step_fsm(state, T_obj, T_tcp)
            state = self._state
            self._T_goal = T_goal
        if isinstance(T_goal, tuple):
            self._publish_viz(self._T_viz, T_obj)
            return self._joint_qp(q, T_tcp, T_goal[1], state)
        self._publish_viz(T_goal, T_obj)
        if T_goal is None:
            return hold
        return self._pose_qp(q, T_tcp, T_goal, v_max, state)

    def _pregrasp_ik(self, T_pre, q_seed):
        """Collision-free IK for the pre-grasp pose, or None.

        Seeds: the current configuration first (short motion), then the ready
        pose (a well-conditioned posture). Joint limits are shrunk by
        ``ik_limit_margin`` so the solution is not right at a limit.
        """
        n = self._num_ctrl_joints
        lo, hi = self._q_lower.copy(), self._q_upper.copy()
        lo[:n] += self._ik_limit_margin
        hi[:n] -= self._ik_limit_margin
        # Seeds: current configuration, the ready pose, and the ready pose
        # turned (joint 1) toward the target, for more distinct solutions.
        q_toward = self._q_home.copy()
        q_toward[0] = np.arctan2(T_pre[1, 3], T_pre[0, 3])
        for seed in (q_seed, self._q_home, q_toward):
            q_sol, _, ok = solve_frame_ik_se3(
                self._pin_model, self._pin_data_plan, seed.copy(), T_pre, self._ik_frame,
                free_joint_idx=np.arange(n), q_lower=lo, q_upper=hi,
                max_iters=self._ik_max_iters, tol=self._ik_tol,
            )
            if not ok or q_sol is None:
                continue
            # Reject solutions near a singularity (e.g. the FR3 shoulder
            # singularity at q2 ~ 0, where joints 1 and 3 line up): the approach
            # from there would need large joint motion for little TCP motion.
            J = get_link_Jacobian(q_sol, self._pin_model, self._pin_data_plan, self._task_link)[:, :n]
            if np.linalg.svd(J, compute_uv=False)[-1] < self._ik_min_sigma:
                continue
            col = self._get_col_data(self._get_col_links(q_sol), self._get_col_obstacles())
            if min(c["dist"] for d in col.values() for c in d.values()) >= self._safety_dist_thr:
                return q_sol
        return None

    def _joint_qp(self, q, T_tcp, q_goal, state) -> PlanResult:
        """Joint-space velocity field toward ``q_goal`` through the same
        collision-aware QP (task Jacobian = identity on the arm joints)."""
        n = self._num_ctrl_joints
        J = np.zeros((n, self._n_x))
        J[:, :n] = np.eye(n)
        x_star = self._reach_joint_gain * (q_goal[:n] - q[:n])
        collision = self._get_col_data(self._get_col_links(q), self._get_col_obstacles())
        qd = self._solve_ik_qp(q, J, x_star, collision)
        return PlanResult(
            qd_des=qd, tip=T_tcp[:3, 3], velocity=np.zeros(3),
            mode=state, at_goal=False, success=True,
        )

    def _step_fsm(self, state, T_obj, T_tcp):
        """Advance the state machine. Returns ``(goal, v_max)``: a TCP pose, a
        ``("joint", q_goal)`` tuple, or ``None`` to hold still."""
        q = self._q_plan
        if state == "reach":
            sel, T_w = self._select(T_obj, T_tcp)
            if sel is None:
                print("[PJGrasp] No valid grasp for this object pose. Stopping.")
                self._enter("idle")
                return None, 0.0
            T_pre = _translate(T_w[sel], [0, 0, -self._pregrasp_dist])
            self._T_viz = T_pre
            if (
                self._ik_for is None
                or self._ik_for[0] != sel
                or np.linalg.norm(T_pre[:3, 3] - self._ik_for[1]) > 0.01
            ):
                self._q_pre = self._pregrasp_ik(T_pre, q)
                if self._q_pre is None:
                    print(f"[PJGrasp] Grasp {sel}: no collision-free IK for the pre-grasp, next one.")
                    self._failed.update((sel, sel ^ 1))
                    self._sel, self._ik_for = None, None
                    return None, 0.0
                self._ik_for = (sel, T_pre[:3, 3].copy())
            n = self._num_ctrl_joints
            if (
                np.max(np.abs(self._q_pre[:n] - q[:n])) < self._reach_joint_tol
                and self._reached(T_tcp, T_pre, self._tol_pre)
            ):
                self._T_obj_at_lock = T_obj.copy()
                self._approach_t0 = time.time()
                self._enter("approach")
            return ("joint", self._q_pre), 0.0

        if state == "approach":
            if np.linalg.norm(T_obj[:3, 3] - self._T_obj_at_lock[:3, 3]) > self._object_moved_tol:
                print("[PJGrasp] Object moved during approach: re-planning.")
                self._enter("reach")
                return None, 0.0
            T_g = T_obj @ self._cand_T[self._sel]
            if (
                time.time() - self._approach_t0 > self._approach_timeout
                or self._stalled(self._T_tcp_meas, T_g)
            ):
                print("[PJGrasp] Approach blocked (fingers touching the object?): "
                      "backing off, trying another grasp.")
                self._failed.update((self._sel, self._sel ^ 1))
                self._back_off(T_tcp)
                return None, 0.0
            # Close only when the reference AND the real TCP are there: if the
            # fingers are pushing on the object the real TCP lags behind.
            actual_err = np.linalg.norm(self._T_tcp_meas[:3, 3] - T_g[:3, 3])
            if self._reached(T_tcp, T_g, self._tol_grasp) and actual_err <= self._actual_pos_tol:
                self._before_close(T_tcp)
                F = self._grip_force(self._sel)
                print(f"[PJGrasp] Grasping: width {self._cand_width[self._sel]:.3f} m, "
                      f"force {F:.1f} N (model needs {self._cand_F[self._sel]:.1f} N)")
                self._wait_for(self._gripper(
                    GripperCmdData.GRASP, self._cand_width[self._sel], F
                ))
                self._enter("close")
                return None, 0.0
            return T_g, self._v_approach

        if state == "close":
            done = self._gripper_done()
            if done is None:
                return None, 0.0
            if done:
                self._T_lift = _translate(np.eye(4), [0, 0, self._lift_height]) @ T_tcp
                self._on_grasped(T_tcp)
                self._enter("lift")
                return None, 0.0
            print("[PJGrasp] Grasp failed: opening, backing off, trying another grasp.")
            self._failed.add(self._sel)
            self._failed.add(self._sel ^ 1)  # its finger-swapped twin
            self._back_off(T_tcp)
            return None, 0.0

        if state in ("lift", "hold"):
            gs = self._gripper_state
            if gs is not None and not gs.is_grasped:
                print("[PJGrasp] Object lost! Opening and starting over.")
                self._back_off(T_tcp)
                return None, 0.0
            if state == "lift":
                if self._reached(T_tcp, self._T_lift, self._tol_pre) or self._stalled(
                    T_tcp, self._T_lift
                ):
                    self._enter("hold")
                    self._on_lifted()
                return self._T_lift, self._v_reach
            return self._T_lift, self._v_reach

        if state == "back_off":
            if self._reached(T_tcp, self._T_backoff, self._tol_pre):
                self._sel, self._ik_for = None, None
                self._enter("reach")
            return self._T_backoff, self._v_approach

        return None, 0.0

    def _back_off(self, T_tcp) -> None:
        self._wait_for(self._gripper(GripperCmdData.MOVE, MAX_WIDTH))
        self._T_backoff = _translate(T_tcp, [0, 0, -self._pregrasp_dist])
        self._enter("back_off")

    def _grip_force(self, sel: int) -> float:
        """Tutorial 10: a fixed force. Tutorial 12 overrides this."""
        return self._grasp_force

    def _on_lifted(self) -> None:
        """Hook: the lift goal was reached."""
        print("[PJGrasp] Lifted and holding. Press x to release.")

    def _before_close(self, T_tcp) -> None:
        """Hook for subclasses (tutorial 12): called just before closing."""

    def _on_grasped(self, T_tcp) -> None:
        """Hook for subclasses (tutorial 12): called when the grasp succeeded,
        before entering ``lift`` (may change ``self._T_lift``)."""

    @staticmethod
    def _reached(T_a, T_b, tol) -> bool:
        """Position within ``tol[0]`` (m) and rotation within ``tol[1]`` (rad)."""
        dp = float(np.linalg.norm(T_b[:3, 3] - T_a[:3, 3]))
        dR = T_a[:3, :3].T @ T_b[:3, :3]
        ang = float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
        return dp <= tol[0] and ang <= tol[1]

    def _pose_qp(self, q, T_tcp, T_goal, v_max, state) -> PlanResult:
        """6-D pose velocity field -> collision-aware IK QP (tutorial 06)."""
        e_p = T_goal[:3, 3] - T_tcp[:3, 3]
        e_R = Rotation.from_matrix(T_goal[:3, :3] @ T_tcp[:3, :3].T).as_rotvec()
        v = _sat(self._pos_gain * e_p, v_max)
        w = _sat(self._rot_gain * e_R, self._w_max)
        J = get_link_Jacobian(q, self._pin_model, self._pin_data_plan, self._task_link)
        # Rows are (angular; linear). Down-weight rotation vs translation.
        W = np.r_[np.full(3, self._rot_weight), np.ones(3)]
        J_task = W[:, None] * J
        x_star = W * np.r_[w, v]
        collision = self._get_col_data(self._get_col_links(q), self._get_col_obstacles())
        qd = self._solve_ik_qp(q, J_task, x_star, collision)
        return PlanResult(
            qd_des=qd, tip=T_tcp[:3, 3], velocity=J[3:] @ qd,
            mode=state, at_goal=False, success=True,
        )

    # ------------------------------------------------------------ control
    def _snapshot_for_plan(self):
        """Plan on the joint *reference* q_ref, not the measured q.

        The IK then steers q_ref kinematically (no dynamics in that loop) and
        the stiff joint spring makes the arm follow q_ref. Planning on the
        measured q instead would feed the arm's small lag behind q_ref (load,
        friction) back into the integrated reference: an integrator around a
        laggy, frictional plant, which ends up in a slow limit cycle.
        """
        q, qd, p_des, mode, init = super()._snapshot_for_plan()
        with self._meas_lock:
            if self._q_ref_shared is not None:
                q[: self._num_ctrl_joints] = self._q_ref_shared
        return q, qd, p_des, mode, init

    def _update(self) -> None:
        """Track the planned joint velocity through an integrated reference.

        ``q_ref += qd_des * dt`` and ``tau = Kp (q_ref - q) + Kd (qd_des - qd)
        + tau_ff``: the stiff spring holds loads (a lifted object) that pure
        velocity damping would let sag, while ``q_ref`` still follows the plan.
        ``q_ref`` is kept within ``max_ref_err`` of ``q`` so it cannot wind up
        when the arm is blocked.
        """
        prev = self._qd_des
        self._consume_latest_plan()
        if self._qd_des is not prev:  # a new plan arrived
            self._plan_stamp = self._cur_time
        with self._meas_lock:
            q = self._q.copy()
            qd = self._qd.copy()
        n = self._num_ctrl_joints

        qd_target = self._qd_des
        if self._cur_time - self._plan_stamp > self._max_plan_age:
            qd_target = np.zeros_like(self._qd_des)
        a = float(np.clip(self._qd_filter, 0.0, 1.0))
        self._qd_des_filt = a * qd_target + (1.0 - a) * self._qd_des_filt
        qd_des = self._qd_des_filt.copy()
        if self._q_ref is None:
            self._q_ref = q[:n].copy()
        self._q_ref = self._q_ref + qd_des[:n] * self._ctrl_dt
        self._q_ref = np.clip(self._q_ref, q[:n] - self._max_ref_err, q[:n] + self._max_ref_err)
        with self._meas_lock:
            self._q_ref_shared = self._q_ref.copy()

        tau_ff = self._tau_ff(q, qd, qd_des)[:n] + self._extra_tau_ff(q)
        cmd = JointCtrlData(num_joints=n)
        cmd.set_data(
            self._cur_time, self._q_ref.copy(), qd_des[:n], tau_ff,
            self._hold_kp[:n], self._hold_kd[:n],
        )
        self.ctrl_pub_que.put(cmd)
        self._publish_target(tip=self._tip, velocity=self._v_cmd)

    def _extra_tau_ff(self, q: np.ndarray) -> np.ndarray:
        """Additional feed-forward torque (tutorial 12: the held object's weight)."""
        return np.zeros(self._num_ctrl_joints)

    # ------------------------------------------------------------ viz out
    def _publish_viz(self, T_goal, T_obj) -> None:
        if self._target_tcp_channel and T_goal is not None:
            msg = SE3PoseData(name=self._target_tcp_channel)
            msg.set_from_matrix(self._cur_time, T_goal)
            self._put_latest(self._target_tcp_channel, msg)
        if self._candidates_channel and T_obj is not None and self._sel is not None:
            a = T_obj[:3, :3] @ self._cand_p[0][self._sel] + T_obj[:3, 3]
            b = T_obj[:3, :3] @ self._cand_p[1][self._sel] + T_obj[:3, 3]
            msg = NamedVecListData(num_vecs=2, vec_dim=3, name=self._candidates_channel)
            msg.set_data(self._cur_time, ["c0_a", "c0_b"], np.vstack([a, b]))
            self._put_latest(self._candidates_channel, msg)

    def _put_latest(self, channel, item) -> None:
        que = self.pub_que_dict.get(channel)
        if que is None:
            return
        while not que.empty():
            try:
                que.get_nowait()
            except Exception:
                break
        try:
            que.put_nowait(item)
        except Exception:
            pass
