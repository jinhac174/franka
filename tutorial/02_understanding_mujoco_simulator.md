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

The model files also include the two finger joints, so the full model (URDF / Pinocchio / MuJoCo) has $9$ coordinates: $[q_{\mathrm{arm}}(7),\, q_{\mathrm{finger1}},\, q_{\mathrm{finger2}}]$. Each finger coordinate is its distance from the center, so the gripper width is $q_{\mathrm{finger1}}+q_{\mathrm{finger2}}$ (0 to 8 cm).

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
| $\tau_{\mathrm{gcomp}}$ | Gravity compensation done by the **robot itself** (see below) |

You do not need to derive this on day one. You *do* need to remember: **the controller never sets $\ddot q$ directly**; it sets $\tau_{\mathrm{act}}$ (through the actuator model), and physics decides the motion.

### The FR3 compensates gravity for you

In torque control, the real FR3’s firmware **always** adds $\tau_{\mathrm{gcomp}} = g(q)$ on top of whatever you command (it knows the arm’s masses and the configured gripper load). It does **not** compensate Coriolis or friction. So on this robot:

$$
\tau_{\mathrm{gcomp}} = g(q)
\quad\Rightarrow\quad
\text{gravity cancels; your command only has to handle everything else.}
$$

The sim reproduces this with `platform.gravity_comp: true` (MuJoCo’s per-body `gravcomp`, which also applies gravity only). That is the default for every tutorial, because it is what the hardware does.

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
| $\tau_{\mathrm{ff}}$ | Model-based torque (Coriolis, contact forces, …) — **not** gravity on the FR3 |

The result is clipped to the FR3 torque limits: 87 Nm for joints 1–4, 12 Nm for joints 5–7.

Special cases you will see often:

- **Gravity compensation (“float”):** $K_p=K_d=0$, $\tau_{\mathrm{ff}}\approx -B\dot q$ (just damping).
- **Velocity field tracking:** $K_p=0$, $K_d>0$, $\dot q_{\mathrm{des}}$ from a planner.

### Double gravity (a common bug)

If the **plant** compensates gravity *and* the controller also sends $g(q)$ in $\tau_{\mathrm{ff}}$, the robot gets **$2g(q)$** and floats upward. Coming from a robot whose controller had to supply $g(q)$, this is the easiest mistake to make on a Franka. Every controller in this repo has a `plant_gravity_comp` flag; it must be `true` for the FR3. [03](03_grav_comp.md) shows what happens when it is wrong.

### The gripper is a different kind of actuator

The Franka Hand runs its own position controller with a force limit. You send it an action:

| Action | Parameters | What happens |
|--------|------------|--------------|
| `MOVE` | width, speed | Fingers go to `width`. Reports failure if something blocks them. |
| `GRASP` | width, speed, force, $\varepsilon_{\mathrm{in}}$, $\varepsilon_{\mathrm{out}}$ | Fingers close until they stop on something, then squeeze with `force`. Success if the final width is within $[\text{width}-\varepsilon_{\mathrm{in}},\ \text{width}+\varepsilon_{\mathrm{out}}]$. |
| `STOP` | – | Stop the current action, release the squeeze. |
| `HOMING` | – | Open fully (on the real hand this also calibrates the max width). |

The sim models the gripper as a MuJoCo position servo on the finger-width tendon with a per-action force cap, and runs the same action logic.

### Contacts vs planning collision models

MuJoCo contacts (meshes, `contype` / `conaffinity`, `solref` / `solimp`) decide **physical** bouncing and friction.

Separately, later tutorials publish **simple FCL shapes** for **planning** ([04](04_publish_collision_geom.md)). Those shapes are *not* required to match MuJoCo’s contact meshes. One is for physics; one is for “stay this far from the wall in the optimizer.”

---

## 3. How it is implemented

| Piece | Where to look | Why it matters |
|-------|---------------|----------------|
| Scene MJCF | `assets/scene/franka_fr3/scene_default.xml` | Floor, walls, lights; includes the robot |
| Robot MJCF | `assets/scene/franka_fr3/fr3_hand_robot.xml` | Masses, damping, armature, friction, joint limits, actuators |
| Robot URDF | `assets/scene/franka_fr3/urdf/fr3_hand.urdf` | The controller’s model (Pinocchio) and the visualizer’s; same kinematics as the MJCF |
| Env base | `env/MujocoBaseEnv.py` | Step loop, real-time vs sim-time |
| Platform | `robot_platform/sim/franka_fr3/SimFrankaFR3.py` | Maps `JointCtrl` → MuJoCo `ctrl`; gravity comp (`modify_mjspec`); gripper emulation (`_GripperSim`) |

When something “feels wrong” (too heavy, too slippery, double gravity), check the MJCF and the `gravity_comp` / `plant_gravity_comp` flags before blaming the controller math.
