"""Helpers for integration tests: start repo processes, listen on LCM."""

import os
import signal
import subprocess
import sys
import time

import lcm

PY = sys.executable


class Proc:
    """A repo process (run_env.py, run_controller.py, ...) in a pseudo-terminal
    (they read keys from the terminal). Send keys with ``key()``; always
    ``stop()`` (done by ``Procs`` as a context manager)."""

    def __init__(self, args, log_path):
        self.log_path = log_path
        cmd = " ".join([PY] + list(args))
        self.p = subprocess.Popen(
            ["script", "-qfec", cmd, "/dev/null"],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            preexec_fn=os.setsid,
        )

    def key(self, keys: str):
        self.p.stdin.write(keys.encode())
        self.p.stdin.flush()

    def log(self) -> str:
        with open(self.log_path, errors="ignore") as f:
            return f.read()

    def stop(self):
        for sig, wait in ((signal.SIGINT, 2.0), (signal.SIGKILL, 0.5)):
            try:
                os.killpg(self.p.pid, sig)
            except ProcessLookupError:
                return
            t0 = time.time()
            while time.time() - t0 < wait and self.p.poll() is None:
                time.sleep(0.05)


class Procs:
    """Context manager that stops every started process, even on failure."""

    def __init__(self, tmp_path):
        self.tmp_path, self.procs = tmp_path, []

    def start(self, name, *args, wait=0.0):
        p = Proc(args, str(self.tmp_path / f"{name}.log"))
        self.procs.append(p)
        time.sleep(wait)
        return p

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for p in reversed(self.procs):
            p.stop()


class Listener:
    """Collect decoded LCM messages: ``last[channel]``, ``count[channel]``."""

    def __init__(self, channels: dict):
        self.lc = lcm.LCM()
        self.last, self.count = {}, {c: 0 for c in channels}
        for ch, decode in channels.items():
            self.lc.subscribe(ch, self._cb(ch, decode))

    def _cb(self, ch, decode):
        def f(_c, data):
            self.count[ch] += 1
            self.last[ch] = decode(data)
        return f

    def spin(self, seconds: float):
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.lc.handle_timeout(10)

    def wait_for(self, pred, timeout: float, step: float = 0.2) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.spin(step)
            if pred():
                return True
        return False

    def reset_counts(self):
        for c in self.count:
            self.count[c] = 0


# --------------------------------------------------------------------------
# Shared bits for the integration tests
# --------------------------------------------------------------------------
ENV = "run_env.py --vis_mode vis_off --config"
CTRL = "run_controller.py --config"
HOME_Q = [0.0, -0.7853982, 0.0, -2.3561945, 0.0, 1.5707963, 0.7853982]
URDF = "assets/scene/franka_fr3/urdf/fr3_robotiq_rigid.urdf"


def decoders():
    """Channel -> decoder for the standard Franka channels."""
    from lcm_type.collision.col_info_t import col_info_t
    from lcm_type.gripper.gripper_state_t import gripper_state_t
    from lcm_type.joint.joint_ctrl_t import joint_ctrl_t
    from lcm_type.joint.joint_meas_t import joint_meas_t
    from lcm_type.pose.se3_pose_t import se3_pose_t

    return {
        "franka_arm_joint_meas": joint_meas_t.decode,
        "franka_arm_joint_ctrl": joint_ctrl_t.decode,
        "franka_gripper_state": gripper_state_t.decode,
        "franka_static_col_info": col_info_t.decode,
        "franka_robot_col_info": col_info_t.decode,
        "franka_object_pose": se3_pose_t.decode,
    }


class TcpFK:
    """TCP position from 7 arm joint angles (Pinocchio, URDF)."""

    def __init__(self):
        import numpy as np
        import pinocchio as pin

        self.np, self.pin = np, pin
        self.m = pin.buildModelFromUrdf(URDF)
        self.d = self.m.createData()
        self.fid = self.m.getFrameId("fr3_gripper_tcp")

    def __call__(self, q7):
        q = self.np.asarray(q7, float)  # the controllers' model: arm only
        self.pin.forwardKinematics(self.m, self.d, q)
        self.pin.updateFramePlacements(self.m, self.d)
        return self.d.oMf[self.fid].translation.copy()


def arm_q(listener):
    import numpy as np

    return np.asarray(listener.last["franka_arm_joint_meas"].q)


def wait_settled(listener, timeout, qd_tol=0.01, hold=1.0):
    """Wait until all arm joint speeds stay below ``qd_tol`` for ``hold`` s."""
    import numpy as np

    t_still = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        listener.spin(0.1)
        m = listener.last.get("franka_arm_joint_meas")
        if m is not None and np.max(np.abs(m.qd)) < qd_tol:
            t_still = t_still or time.time()
            if time.time() - t_still > hold:
                return True
        else:
            t_still = None
    return False

# Process start-up (imports, model compile, first messages). Generous: on a
# loaded machine or slow network filesystem this can take far longer than the
# usual few seconds; wait_for returns as soon as the condition holds.
STARTUP = 180
