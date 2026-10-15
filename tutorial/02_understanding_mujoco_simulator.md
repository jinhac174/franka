# Understanding the MuJoCo simulator

This tutorial explains the **physics side** of the stack: what MuJoCo integrates, how your LCM command becomes torque, and why timing matters. You do not write new code here — you learn to *read* the plant that every controller talks to.

If [01](01_quick_start.md) was “who talks to whom,” this chapter is “what is being controlled.”

---

## 1. Run & LCM communication

Use the same three commands as the quick start:

```bash
python run_env.py --config configs/franka_fr3/env/default.yaml

python run_controller.py --config configs/franka_fr3/controllers/grav_comp.yaml

python run_visualizer.py --config configs/visualizer/franka_fr3/default.yaml
```

No new channels. While it runs, keep this picture in mind:

- **Controller → env:** `franka_arm_joint_ctrl` (desired motion + feedforward torque + gains, 7 arm joints) and `franka_gripper_cmd` (gripper actions).
- **Env → controller:** joint measurements on the `franka_*_joint_meas` channels, gripper status on `franka_gripper_state`.

### Timing knobs in the env YAML

| Knob | Intuition |
|------|-----------|
| `sim_freq: 1000` | Physics tries to take 1000 steps per simulated second ($h=1\,\mathrm{ms}$) — the same 1 kHz the real FR3 control loop runs at. |
| `sim_steps_per_control` | How many physics steps share **one** held motor command. |
| `env_mode: sim_real_time` | Do not run faster than wall-clock (good for interactive demos). |
| `env_mode: sim_sim_time` | Run as fast as the CPU allows (good for batch tests). |

---

## 2. Ideas (the plant model)

### Coordinates

The arm’s configuration is a vector of joint angles

$$
q \in \mathbb{R}^{7}.
$$

The gripper, a **Robotiq 2F-85**, is a linkage: on each side a driver, a spring link, a coupler and a follower carrying the pad, all following **one** coordinate, the driver angle (0 = open, about 0.785 rad = closed). The pads stay parallel and the opening (the width between the pad faces) goes from 85 mm to 0. Three models describe it:

| Model | Used by | Gripper |
|---|---|---|
| MuJoCo `fr3_robotiq_robot.xml` | the simulator | the full linkage, closed by equality constraints, driven by one position servo |
| `urdf/fr3_robotiq.urdf` | Viser | one joint, `robotiq_driver_joint`; the other links follow it (`<mimic>`) |
| `urdf/fr3_robotiq_rigid.urdf` | the controllers (Pinocchio) | rigid and open: a payload on the flange, so $q$ is the 7 arm joints |

`robot_platform/gripper/robotiq_2f85.py` converts between the driver angle and the width.

Velocities are $\dot q$, accelerations $\ddot q$. MuJoCo advances $(q,\dot q)$ through time using the equations of motion below.

### Equations of motion (intuition first)

Newton’s law for a multi-joint robot is more than $F=ma$. Inertia depends on pose, joints couple through Coriolis forces, gravity pulls on every link, friction dissipates energy, and contacts push when you hit the table.

A standard form is:

$$
\bigl(M(q)+\mathrm{diag}(I_a)\bigr)\ddot q
  + C(q,\dot q)\dot q
  + g(q)
  + \tau_{\mathrm{damp}}
  + \tau_{\mathrm{fric}}
  + \tau_{\mathrm{contact}}
  = \tau_{\mathrm{act}} + \tau_{\mathrm{gcomp}}.
$$

| Symbol | Plain-language meaning |
|--------|-------------------------|
| $M(q)$ | Inertia matrix (“how hard it is to accelerate each joint”) |
| $I_a$ | Rotor / gearbox inertia (`armature` in MJCF) |
| $C\dot q$ | Velocity-dependent coupling (Coriolis / centrifugal) |
| $g(q)$ | Gravity torques |
| $\tau_{\mathrm{damp}}$ | Viscous friction $\approx -b\dot q$ |
| $\tau_{\mathrm{fric}}$ | Coulomb-like friction (`frictionloss`) — the FR3 joints have about 0.2–1.1 Nm |
| $\tau_{\mathrm{contact}}$ | Soft contact / constraint forces from MuJoCo |
| $\tau_{\mathrm{act}}$ | **What your controller asks the motors to produce** |
| $\tau_{\mathrm{gcomp}}$ | Gravity compensation done by the **plant itself**: 0 in this repo's default (see below) |

You do not need to derive this on day one. You *do* need to remember: **the controller never sets $\ddot q$ directly**; it sets $\tau_{\mathrm{act}}$ (through the actuator model), and physics decides the motion.

### Who compensates gravity

In torque control, the real FR3’s firmware **always** adds its own $g(q)$ on top of whatever you command (from its model of the arm and the gripper load configured in Desk). It does **not** compensate Coriolis or friction.

This repo does not rely on it. As in the original tutorials, **the controller computes $g(q)$** from its URDF and sends it in $\tau_{\mathrm{ff}}$; the plant adds nothing:

- sim: `platform.gravity_comp: false` (MuJoCo applies plain gravity);
- robot: `run_hardware.py` with `gravity_model: controller` subtracts the FR3's own $g(q)$ every tick ([15](15_real_fr3.md)).

$$
\tau_{\mathrm{gcomp}} = 0
\quad\Rightarrow\quad
\text{the controller's } g(q) \text{ holds the arm, in the sim and on the robot alike.}
$$

`platform.gravity_comp: true` (MuJoCo’s per-body `gravcomp`) reproduces the FR3 firmware instead; [03](03_grav_comp.md) compares the two.

### How `JointCtrl` becomes $\tau_{\mathrm{act}}$

Inside the platform, the actuator law is a familiar PD + feedforward form, computed for the 7 arm joints:

$$
\tau_{\mathrm{act}}
  = K_p\,(q_{\mathrm{des}}-q)
  + K_d\,(\dot q_{\mathrm{des}}-\dot q)
  + \tau_{\mathrm{ff}}.
$$

| Term | When it is used |
|------|----------------------|
| $K_p(q_{\mathrm{des}}-q)$ | Spring toward a joint pose (impedance, tracking, …) |
| $K_d(\dot q_{\mathrm{des}}-\dot q)$ | Damper / velocity tracking |
| $\tau_{\mathrm{ff}}$ | Model-based torque: gravity $g(q)$, Coriolis, contact forces, … |

The result is clipped to the FR3 torque limits: 87 Nm for joints 1–4, 12 Nm for joints 5–7.

Special cases you will see often:

- **Gravity compensation (“float”):** $K_p=K_d=0$, $\tau_{\mathrm{ff}}= g(q) -B\dot q$.
- **Velocity field tracking:** $K_p=0$, $K_d>0$, $\dot q_{\mathrm{des}}$ from a planner.

### Double gravity (a common bug)

If the **plant** compensates gravity *and* the controller also sends $g(q)$ in $\tau_{\mathrm{ff}}$, the robot gets **$2g(q)$** and floats upward; if neither does, it falls. Every controller has a `plant_gravity_comp` flag (here `false`: the controller sends $g(q)$) that must match the plant (`platform.gravity_comp` in the sim, `gravity_model` on the robot). [03](03_grav_comp.md) shows what happens when they do not match.

### The gripper is a different kind of actuator

The Robotiq 2F-85 runs its own position controller with a force limit (on the real robot it is driven over USB, Modbus RTU, as in DROID). You send it an action, in metres and newtons:

| Action | Parameters | What happens |
|--------|------------|--------------|
| `MOVE` | width, speed | Fingers go to `width`. Reports failure if something blocks them. |
| `GRASP` | width, speed, force, $\varepsilon_{\mathrm{in}}$, $\varepsilon_{\mathrm{out}}$ | Fingers close until they stop on something, then squeeze with `force`. Success if the final width is within $[\text{width}-\varepsilon_{\mathrm{in}},\ \text{width}+\varepsilon_{\mathrm{out}}]$. |
| `STOP` | – | Stop the current action, release the squeeze. |
| `HOMING` | – | Activate and open fully (on the real gripper this also measures its open / closed positions). |

Ranges: 0–85 mm, 20–150 mm/s, 20–235 N. The sim models the gripper as a MuJoCo position servo on the driver angle with a per-action force cap (converted from the squeeze force at the pads), and runs the same action logic.

### Contacts vs planning collision models

MuJoCo contacts (meshes, `contype` / `conaffinity`, `solref` / `solimp`) decide **physical** bouncing and friction.

Separately, later tutorials publish **simple FCL shapes** for **planning** ([04](04_publish_collision_geom.md)). Those shapes are *not* required to match MuJoCo’s contact meshes. One is for physics; one is for “stay this far from the wall in the optimizer.”

---

## 3. How it is implemented

| Piece | Where to look | Why it matters |
|-------|---------------|----------------|
| Scene MJCF | `assets/scene/franka_fr3/scene_default.xml` | Floor, walls, lights; includes the robot |
| Robot MJCF | `assets/scene/franka_fr3/fr3_robotiq_robot.xml` | Masses, damping, armature, friction, joint limits, actuators |
| Robot URDF | `assets/scene/franka_fr3/urdf/fr3_robotiq_rigid.urdf` | The controller’s model (Pinocchio) and the visualizer’s; same kinematics as the MJCF |
| Env base | `env/MujocoBaseEnv.py` | Step loop, real-time vs sim-time |
| Platform | `robot_platform/sim/franka_fr3/SimFrankaFR3.py` | Maps `JointCtrl` → MuJoCo `ctrl`; gravity comp (`modify_mjspec`); gripper emulation (`_GripperSim`) |

When something “feels wrong” (too heavy, too slippery, double gravity), check the MJCF and the `gravity_comp` / `plant_gravity_comp` flags before blaming the controller math.
