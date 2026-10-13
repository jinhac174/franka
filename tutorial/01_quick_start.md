# Quick start

Welcome to RoPhiTutorial. This first lesson is not about fancy control theory — it is about **how the pieces talk to each other**. Once you can start the simulation and see the robot move under a simple controller, every later tutorial reuses the same pattern.

We use a **Franka Research 3 (FR3)** arm (7 joints) with a **Franka Hand** parallel gripper (2 coupled fingers), simulated in MuJoCo, controlled in a separate Python process, and visualized in Viser. The same processes and messages later drive the real FR3.

---

## 1. Run & LCM communication

### What you will launch

Open **three terminals** in the repository root and run:

```bash
python run_env.py --config configs/franka_fr3/env/default.yaml

python run_controller.py --config configs/franka_fr3/controllers/grav_comp.yaml

python run_visualizer.py --config configs/visualizer/franka_fr3/default.yaml
```

Start the **env** first (it owns the physics clock), then the controller, then the visualizer. You should see a MuJoCo window and a Viser page (http://localhost:8080) with the arm in Franka's “ready” pose; under gravity compensation it holds still and can be pushed around ([03](03_grav_comp.md)).

| Key | Where | Action |
|-----|--------|--------|
| `p` | env or controller terminal | Pause |
| `q` | env or controller terminal | Quit |

### Why three processes?

| Process | Everyday analogy | Job |
|---------|------------------|-----|
| **Env** | The physical world | Advance MuJoCo; publish “what the sensors see”; apply motor commands |
| **Controller** | The brain / real-time computer | Read sensors; compute torques / setpoints; publish commands |
| **Visualizer** | A camera + HUD | Draw the robot for you (does not affect physics) |

Keeping them separate means:

- physics can run at 1000 Hz while graphics runs at 30 Hz,
- a crash in the visualizer does not kill the controller,
- the same controller talks to the **real** FR3 over the same message types — only the env process is swapped for the hardware process.

### What is LCM?

**LCM** (Lightweight Communications and Marshalling) is a publish/subscribe bus. Think of named **channels** (like radio stations):

- one process **publishes** a message on a channel,
- any process that **subscribes** receives a copy.

You do not call the other process’s Python functions directly. You only exchange messages. That is the discipline of this codebase.

### Channels in this tutorial

| Channel name | Message type | Who → whom | Meaning in plain words |
|--------------|--------------|------------|------------------------|
| `franka_arm_joint_meas` | `JointMeas` | env → controller & viz | Arm joint angles $q$, velocities $\dot q$, torques $\tau$ (7 each), 1000 Hz |
| `franka_gripper_joint_meas` | `JointMeas` | env → controller & viz | The two finger positions (each = gripper width / 2), 100 Hz |
| `franka_gripper_state` | `GripperState` | env → controller | Width, `is_grasped`, `busy`, result of the last gripper command |
| `franka_arm_joint_ctrl` | `JointCtrl` | controller → env | What the 7 arm motors should do next (see below) |
| `franka_gripper_cmd` | `GripperCmd` | controller → env | Gripper action: move / grasp / stop / homing |
| `sim_clock` | time | env → controller | Simulation time so rates stay synchronized |

A `JointCtrl` message packages five vectors of length 7 (one entry per arm joint):

$$
\bigl(
  q_{\mathrm{des}},\;
  \dot q_{\mathrm{des}},\;
  \tau_{\mathrm{ff}},\;
  K_p,\;
  K_d
\bigr).
$$

You will learn what each means in [02](02_understanding_mujoco_simulator.md). For now: the env turns that message into motor torques.

**Why is the gripper separate?** The arm is torque controlled at 1 kHz. The real Franka Hand is not: it only accepts high-level actions like “move to 4 cm at 0.1 m/s” or “grasp with 40 N”, which its own controller carries out. So the gripper gets its own command and state channels, and the sim reproduces exactly that behavior. Grasping tutorials ([10](10_reactive_grasping.md)+) use them.

Later tutorials add more channels (collision shapes, object poses, arrows). The **pattern** never changes: env publishes world state; controller publishes commands; visualizer listens.

```
                 measurements
        ┌──────────────────────────► Controller
        │                                │
Env ────┤                                │  JointCtrl (+ GripperCmd)
        │                                ▼
        │◄───────────────────────────────┘
        │
        └────── measurements ──────────► Visualizer
```

---

## 2. Ideas (architecture)

### The control loop in words

Imagine repeating forever, about once every millisecond:

1. **Sense.** The env reads the current joint angles from MuJoCo and publishes them.
2. **Think.** The controller computes a new command from those angles (here: just a little damping — see [03](03_grav_comp.md)).
3. **Act.** The env applies that command the next time it steps the physics.

This is a **feedback loop**: the command depends on the measurement, which depends on how the robot moved under the previous command.

### Why not one big Python script?

A single script that steps MuJoCo, runs OSQP, and draws OpenGL in one thread quickly becomes slow and brittle. Separating rates and responsibilities is standard on real robots (and in this course).

### Zero-order hold (gentle version)

The controller and the simulator may not wake up at the exact same instant. Between controller updates, the env **holds the last command constant**. That is called a zero-order hold. It is ordinary and expected; just remember that a slow controller means the robot “coasts” on an old command for longer.

On the real FR3 the same thing happens inside the hardware process: it re-sends the latest command to the robot every millisecond, and stops the robot if commands stop arriving.

---

## 3. How it is implemented

You do not need to read all of this on day one. Use it as a map.

| Piece | Path | What it does |
|-------|------|----------------|
| Env entry | `run_env.py` | Loads YAML, builds the MuJoCo env, starts LCM pubs/subs |
| Controller entry | `run_controller.py` | Loads YAML, builds a controller by `name`, starts the loop |
| Visualizer entry | `run_visualizer.py` | Loads YAML, opens Viser, subscribes to measurements |
| Robot platform (sim) | `robot_platform/sim/franka_fr3/SimFrankaFR3.py` | Turns `JointCtrl` into motor torques; emulates the gripper; publishes measurements |
| Default controller | `controller/GravCompControl.py` | Gravity compensation used in this quick start |
| Env config | `configs/franka_fr3/env/default.yaml` | Scene file, `sim_freq`, channel names |
| Controller config | `configs/franka_fr3/controllers/grav_comp.yaml` | `ctrl_freq`, URDF, which meas channels map to which joints |
| Viz config | `configs/visualizer/franka_fr3/default.yaml` | URDF for drawing, same meas channels |

**Mental model:** YAML chooses *which* classes and *which* channel strings; Python classes implement *behavior*; LCM carries *numbers* between processes.

Any YAML value can be overridden from the command line with dotted keys, e.g. `python run_env.py --config ... --vis_mode vis_off --platform.gravity_comp false`.

When you are comfortable starting and stopping this stack, continue to [02](02_understanding_mujoco_simulator.md) (what MuJoCo is doing) and [03](03_grav_comp.md) (your first real control law).
