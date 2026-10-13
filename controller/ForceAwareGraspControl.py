"""Force-aware parallel-jaw pick for the FR3 + Franka Hand (tutorial 12).

Extends tutorial 10's reactive grasp with "weigh, then squeeze":

1. Just before closing (arm at rest, gripper empty) record the external joint
   torque ``tau_ext = tau_J - g(q)`` as a *tare*.
2. Grasp with a firm probe force and lift a little (``test_lift_height``).
   If the estimate below says ~0 kg, the object slipped: back off, retry harder.
3. Settle, average ``tau_ext`` again. The change ``dtau`` is the torque the
   arm spends holding the object: ``dtau = J^T w``, so least squares gives the
   load wrench ``w`` at the TCP and the object mass ``m = f_z / g``.
4. Required grip force from the tutorial-11 model for the selected grasp,
   scaled by the estimated mass, times ``safety_factor``. Re-grasp with it.
5. Add ``J^T w`` as feed-forward torque (the FR3 only compensates its own and
   the gripper's weight), then lift fully.
6. If the object is lost anyway, the next attempt probes with
   ``escalation`` times more force.

On the real FR3 the torque sensors give ``tau_J`` and libfranka also offers
``tau_ext_hat_filtered`` directly; the sim publishes the same link-side torque.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import pinocchio as pin
from omegaconf import DictConfig

from controller.ParallelJawGraspControl import ParallelJawGraspController, _translate
from data_type.basic_types.GripperCmdData import GripperCmdData
from utils.pinocchio.getter import get_link_Jacobian

G = 9.81


class ForceAwareGraspController(ParallelJawGraspController):
    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        f = config["force"]
        self._f_probe = float(f.get("probe_force", 30.0))
        self._min_mass = float(f.get("min_mass", 0.05))
        self._safety = float(f.get("safety_factor", 2.5))
        self._f_min = float(f.get("f_min", 5.0))
        self._f_max = float(f.get("f_max", 70.0))
        self._escalation = float(f.get("escalation", 1.5))
        self._test_lift = float(f.get("test_lift_height", 0.02))
        self._settle_time = float(f.get("settle_time", 0.5))
        self._weigh_time = float(f.get("weigh_time", 0.5))
        self._feedforward = bool(f.get("load_feedforward", True))

        # Grip forces in _cand_F were computed for this mass (scale linearly).
        self._mass_at_build = self._obj_mass
        self._phase = "probe"  # probe -> final
        self._force_scale = 1.0
        self._f_cmd = None
        self._tare = None
        self._w_load = None  # load wrench (angular; linear) at the TCP, world frame
        self._hold_t0 = 0.0
        # (time, q_arm, tau_J) samples from the 1 kHz loop
        self._samples: deque = deque(maxlen=int(4.0 * self._ctrl_freq))

    # ------------------------------------------------------------ sampling
    def _update(self) -> None:
        """Control tick (1 kHz): as tutorial 10, plus record (q, tau_J).

        Sample here, not in ``_check_and_get_data_from_que``: that one runs on
        every spin of the controller's busy loop (several kHz), and taking the
        lock that often starves the planning thread.
        """
        super()._update()
        n = self._num_ctrl_joints
        with self._meas_lock:
            self._samples.append((time.time(), self._q[:n].copy(), self._tau[:n].copy()))

    def _tau_ext(self, window: float):
        """Mean of ``tau_J - g(q)`` over the last ``window`` seconds, + mean q."""
        t_now = time.time()
        with self._meas_lock:
            rows = [(q, tau) for t, q, tau in self._samples if t_now - t <= window]
        if len(rows) < 10:
            return None, None
        q = np.mean([r[0] for r in rows], axis=0)
        tau = np.mean([r[1] for r in rows], axis=0)
        q_full = self._q_home.copy()
        q_full[: len(q)] = q
        g = pin.computeGeneralizedGravity(self._pin_model, self._pin_data_plan, q_full)
        return tau - g[: len(q)], q_full

    # ------------------------------------------------------------ hooks
    def _before_close(self, T_tcp) -> None:
        if self._phase == "probe":
            self._tare, _ = self._tau_ext(0.3)

    def _grip_force(self, sel: int) -> float:
        if self._phase == "probe":
            return min(self._f_probe * self._force_scale, self._f_max)
        return self._f_cmd

    def _on_grasped(self, T_tcp) -> None:
        if self._phase == "probe":
            self._T_lift = _translate(np.eye(4), [0, 0, self._test_lift]) @ T_tcp

    def _on_lifted(self) -> None:
        if self._phase == "probe":
            print(f"[ForceGrasp] Test lift ({self._test_lift * 100:.0f} cm): settling, then weighing...")
        else:
            super()._on_lifted()

    def _enter(self, state: str) -> None:
        if state == "hold" and self._state != "hold":
            self._hold_t0 = time.time()
        super()._enter(state)

    def _back_off(self, T_tcp) -> None:
        if self._state in ("lift", "hold", "regrasp"):
            # The object was in the hand and got lost: squeeze harder next time.
            self._force_scale *= self._escalation
            print(f"[ForceGrasp] Next probe force x{self._force_scale:.2f}.")
        self._phase = "probe"
        self._w_load = None
        super()._back_off(T_tcp)

    def _handle_key(self, key: str) -> None:
        if key == "v":
            self._phase, self._force_scale, self._w_load = "probe", 1.0, None
        if key == "x":
            self._w_load = None
        super()._handle_key(key)

    # ------------------------------------------------------------ FSM
    def _step_fsm(self, state, T_obj, T_tcp):
        if state == "hold" and self._phase == "probe":
            gs = self._gripper_state
            if gs is not None and not gs.is_grasped:
                return super()._step_fsm(state, T_obj, T_tcp)  # lost handling
            if time.time() - self._hold_t0 < self._settle_time + self._weigh_time:
                return self._T_lift, self._v_reach
            self._weigh_and_regrasp()
            return self._T_lift, self._v_reach

        if state == "regrasp":
            done = self._gripper_done()
            if done is None:
                return self._T_lift, self._v_reach
            if not done:
                print("[ForceGrasp] Re-grasp failed.")
                self._back_off(T_tcp)
                return None, 0.0
            self._phase = "final"
            self._T_lift = _translate(np.eye(4), [0, 0, self._lift_height - self._test_lift]) @ T_tcp
            self._enter("lift")
            return self._T_lift, self._v_reach

        return super()._step_fsm(state, T_obj, T_tcp)

    def _weigh_and_regrasp(self) -> None:
        tau_ext, q = self._tau_ext(self._weigh_time)
        if tau_ext is None or self._tare is None:
            print("[ForceGrasp] Not enough torque samples; keeping the probe force.")
            self._f_cmd = self._grip_force(self._sel)
        else:
            dtau = tau_ext - self._tare
            n = self._num_ctrl_joints
            J = get_link_Jacobian(q, self._pin_model, self._pin_data_plan, self._task_link)[:, :n]
            w, *_ = np.linalg.lstsq(J.T, dtau, rcond=None)  # dtau = J^T w
            m_est = float(w[5] / G)
            if m_est < self._min_mass:
                # The scale reads ~0: the object slid out / stayed on the table
                # even though the fingers still look closed.
                print(f"[ForceGrasp] Weighed {m_est:.3f} kg: the object did not come up "
                      "(slipped). Opening, squeezing harder next time.")
                self._back_off(self._T_lift)
                return
            F_req = float(self._cand_F[self._sel]) * max(m_est, 0.0) / self._mass_at_build
            self._f_cmd = float(np.clip(self._safety * F_req * self._force_scale, self._f_min, self._f_max))
            self._w_load = w if self._feedforward else None
            print(
                f"[ForceGrasp] Weighed: m_est = {m_est:.3f} kg (env says {self._obj_mass:.3f} kg). "
                f"Model needs {F_req:.1f} N -> re-grasp with {self._f_cmd:.1f} N "
                f"(safety x{self._safety:g}"
                + (f", escalation x{self._force_scale:.2f}" if self._force_scale > 1.0 else "")
                + ")."
            )
        # Re-grasp around the *measured* width: grasp() first moves toward the
        # target width, so a target slightly wider than the object would open
        # the fingers and drop it. Slightly narrower = keep closing on it.
        gs = self._gripper_state
        width = (gs.width if gs is not None else self._cand_width[self._sel]) - 0.002
        self._wait_for(self._gripper(GripperCmdData.GRASP, width, self._f_cmd))
        self._enter("regrasp")

    # ------------------------------------------------------------ control
    def _extra_tau_ff(self, q: np.ndarray) -> np.ndarray:
        """Hold the object's weight with feed-forward: tau = J(q)^T w_load."""
        w = self._w_load
        n = self._num_ctrl_joints
        if w is None:
            return np.zeros(n)
        J = get_link_Jacobian(q, self._pin_model, self._pin_data_ctrl, self._task_link)[:, :n]
        return J.T @ w
