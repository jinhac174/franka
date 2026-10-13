# Gravity compensation

Your first real control law: make the arm feel **weightless** (almost), without telling it *where* to go.

On the FR3 this is special: the robot already compensates gravity itself in torque mode ([02](02_understanding_mujoco_simulator.md)). So this tutorial does it twice — once the way the FR3 runs, and once in a **sim-only** mode where *you* compute $g(q)$, so you see exactly what the firmware does for you.

---

## 1. Run & LCM communication

### A. The FR3 way (plant compensates gravity)

```bash
python run_env.py --config configs/franka_fr3/env/default.yaml

python run_controller.py --config configs/franka_fr3/controllers/grav_comp.yaml

python run_visualizer.py --config configs/visualizer/franka_fr3/default.yaml
```

### B. Sim-only: the controller computes $g(q)$

Turn the plant’s compensation off and the controller’s on:

```bash
python run_env.py --config configs/franka_fr3/env/default.yaml --platform.gravity_comp false

python run_controller.py --config configs/franka_fr3/controllers/grav_comp.yaml --plant_gravity_comp false

python run_visualizer.py --config configs/visualizer/franka_fr3/default.yaml
```

This mode does not exist on the real FR3 (its gravity compensation cannot be disabled). It is here so you can see the math working.

| Key | Action |
|-----|--------|
| `p` / `q` | Pause / quit |

### What you should observe

In both A and B the arm **holds its pose against gravity** but does **not** spring back to a home configuration. Drag a link in the MuJoCo viewer (double-click a body, then Ctrl + right-drag) and let go: it stays roughly where you left it. The FR3 joints have noticeable friction, so small pushes may not move it at all.

In B the arm may drift very slightly (on the order of a milliradian): the URDF the controller uses and the MuJoCo model differ a little in the gripper’s mass. That is the *model mismatch* discussed below.

### Try: double gravity

Run the env as in **A** but the controller as in **B** (`--plant_gravity_comp false` only). Now gravity is cancelled twice; the arm drifts upward (about 0.3 rad in the first 3 s). If you ever see a Franka “float up” when a controller starts, this is the first thing to check.

### LCM (same as quick start)

| Channel | Direction | Role here |
|---------|-----------|-----------|
| `franka_arm_joint_meas` | env → ctrl | Arm $q,\dot q$ |
| `franka_gripper_joint_meas` | env → ctrl | Finger positions (completes the 9-joint model for Pinocchio) |
| `franka_arm_joint_ctrl` | ctrl → env | A: $\tau_{\mathrm{ff}}=-B\dot q$. B: $\tau_{\mathrm{ff}}=g(q)-B\dot q$. Always $K_p=K_d=0$ |

---

## 2. Ideas (the algorithm)

### The problem in one sentence

Gravity produces joint torques $g(q)$ that depend on the configuration. If nothing cancels them, the arm collapses. If the motors add **exactly** $g(q)$, those gravity torques cancel and the arm “floats.”

### Feedforward gravity

We compute $g(q)$ from the URDF model (Pinocchio’s `computeGeneralizedGravity`) and send it as feedforward:

$$
\tau_{\mathrm{ff}} = g(q) - B_{\mathrm{arm}}\dot q_{\mathrm{arm}}.
$$

On the FR3 the first term is supplied by the robot, so the controller sends only

$$
\tau_{\mathrm{ff}} = - B_{\mathrm{arm}}\dot q_{\mathrm{arm}}.
$$

Either way, we set PD gains to zero and do not invent a goal pose:

$$
K_p = 0,\qquad K_d = 0,\qquad
q_{\mathrm{des}} = q,\qquad \dot q_{\mathrm{des}} = 0.
$$

So the actuator law from [02](02_understanding_mujoco_simulator.md) reduces to $\tau_{\mathrm{act}}=\tau_{\mathrm{ff}}$.

The small term $B_{\mathrm{arm}}\dot q_{\mathrm{arm}}$ is **extra damping on the arm**. It bleeds kinetic energy so the arm does not swing forever.

### Why it settles (energy picture)

With matched models, the closed loop looks like

$$
M(q)\ddot q + C(q,\dot q)\dot q + B\dot q = 0.
$$

A standard energy argument shows that kinetic energy

$$
E = \tfrac12 \dot q^\top M(q)\dot q
$$

is nonincreasing when $B$ is positive definite on the moving joints:

$$
\dot E = -\dot q^\top B\dot q \le 0.
$$

So the robot coasts and eventually stops. Importantly, **every configuration with $\dot q=0$ is an equilibrium**. Gravity compensation does **not** create a preferred home pose — that is the job of impedance or tracking controllers later.

### Model mismatch (gentle warning)

If the model's masses differ from reality, residual gravity remains and the arm slowly drifts. In mode B you see this directly (URDF vs MuJoCo gripper mass). On the real FR3 the same issue shows up as the **end-effector load**: the firmware compensates the arm plus whatever gripper mass is configured in Franka Desk. If you attach something heavier and do not update that setting, the arm sags. Treat perfect floating as a modeling assumption, not magic.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/GravCompControl.py` | Each control tick: read $q$, compute $g(q)$ unless `plant_gravity_comp`, subtract arm damping, publish `JointCtrl` for the 7 arm joints |
| `configs/franka_fr3/controllers/grav_comp.yaml` | `ctrl_freq`, URDF path, `plant_gravity_comp`, `arm_damping`, joint↔channel map |
| `SimFrankaFR3.modify_mjspec` | How the sim reproduces the FR3’s own gravity compensation |

**Reading tip:** find where `kp` and `kd` are set to zero. That single choice is what distinguishes “float” from “servo to a pose.”

Next: [04](04_publish_collision_geom.md) adds geometry messages; [05](05_collision_aware_grav_comp.md) uses them so floating does not drive through walls.
