# Running on the real FR3

Everything so far ran in MuJoCo. On the robot, exactly one process changes: `run_env.py` is replaced by `run_hardware.py`, which talks to the FR3 and the Franka Hand and publishes **the same LCM channels**. Controllers, visualizer and perception run unchanged; controllers only need `--env_mode real` (they then use wall-clock time instead of the simulation clock).

```
                 sim                                   real robot
run_env.py  (MuJoCo, SimFrankaFR3)     ->     run_hardware.py (panda-py -> FR3, Franka Hand)
                                              run_scene_info.py (collision geometry of your cell)
run_controller.py --config ...         ->     run_controller.py --config ... --env_mode real
run_visualizer.py --config ...         ->     (unchanged)
```

---

## 1. Workstation setup (once)

Follow Franka’s documentation for these; they depend on your robot’s system version:

1. **Real-time kernel.** libfranka needs a `PREEMPT_RT` kernel for its 1 kHz loop.
2. **libfranka version** compatible with your FR3’s system version (Franka publishes a compatibility table).
3. **panda-py built for that libfranka version.** The PyPI wheel (`pip install panda-python`) bundles libfranka 0.9.2, which only supports the older Panda. Install the FR3 build from the panda-py GitHub releases instead (see `requirements.txt`).
4. **Network and Desk.** Connect the workstation to the control box, unlock the joints and activate FCI in Desk. Set the gripper’s mass / inertia in Desk if you changed it: the FR3’s gravity compensation uses it.
5. **LCM** multicast and receive buffer, as in [13](13_foundation_pose_sim.md).

Then set `robot.hostname` in `configs/franka_fr3/hardware/fr3.yaml` and check the machine:

```bash
python scripts/preflight.py             # RT kernel, rtprio, panda-py / libfranka, LCM, config, ping
python scripts/preflight.py --connect   # + connect with panda-py and read the state (no motion)
pytest -m "not integration"             # the repo works on this machine
```

---

## 2. What `run_hardware.py` does

Every ~1 ms it reads the robot state, takes the newest `JointCtrl` command and computes, exactly like the simulator,

$$
\tau = K_p(q_{\mathrm{des}}-q) + K_d(\dot q_{\mathrm{des}}-\dot q) + \tau_{\mathrm{ff}},
$$

and hands it to panda-py’s `AppliedTorque` controller, which feeds libfranka’s 1 kHz loop (holding the latest value). The FR3 adds gravity compensation on top — the reason all controllers use `plant_gravity_comp: true`. Gripper commands run in their own thread (`move` / `grasp` block until done); `stop` interrupts them.

### Safety layer

On top of the robot’s own reflexes and the **user stop** (keep it in your hand):

| Mechanism | Config (`safety:`) | Why |
|-----------|-------------------|-----|
| Joint-error clamp | `max_joint_error` (0.1 rad) | $q_{\mathrm{des}}$ is clamped to $q \pm 0.1$: a controller that jumps its set-point far away (tutorial 09, key `3`) only gets a bounded spring force |
| Torque limits | `tau_max` | Never above the FR3 limits |
| Torque-rate limit | `tau_rate_max` (1000 Nm/s) | No torque steps (libfranka rejects them, and they jerk the arm) |
| Command timeout | `ctrl_timeout` (0.1 s) | Controller stopped / crashed / not started: hold the pose |
| Speed guard | `max_joint_vel` | Above it: hold until you press `h` |

Keys in the hardware terminal: `q` quit (stops torque control), `h` clear a speed stop, `r` recover after a robot error (reflex), then torque control restarts.

Tested here only against a fake robot (`--robot.backend fake`); see section 4 for how to verify on yours.

---

## 3. First session, step by step

Increase risk one step at a time, with the user stop in hand:

1. **Dry run without the robot** on the workstation:
   ```bash
   python run_hardware.py --config configs/franka_fr3/hardware/fr3.yaml --robot.backend fake
   python run_controller.py --config configs/franka_fr3/controllers/grav_comp.yaml --env_mode real
   ```
2. **Robot, no controller.** `run_hardware.py` alone: the arm should hold still (hold mode).
3. **Gravity compensation** ([03](03_grav_comp.md)): start the grav-comp controller with `--env_mode real`. Hand-guide the arm: it floats. If it drifts, check the gripper load in Desk.
4. **Visualizer:** `run_visualizer.py --config configs/visualizer/franka_fr3/default.yaml` should mirror the real arm.
5. **Impedance** ([08](08_impedance_control.md)), joint mode first, with lowered gains (`--kq_scale 0.3`).
6. **Collision-aware controllers** ([05](05_collision_aware_grav_comp.md)–[07](07_computed_torque_control.md), [09](09_inverse_kinematics.md)): first describe your cell — walls, table — in a scene XML (`floor`, `wall_*`, `table*`, `obs_*` geoms) and publish it:
   ```bash
   python run_scene_info.py --config configs/franka_fr3/hardware/scene_info.yaml
   ```
   The default scene’s walls are placeholders.
7. **Gripper:** the grasping controllers send real `move` / `grasp` commands.
8. **Grasping with perception** (section 6).

`scripts/launch.sh <target> real` starts the processes of a step in tmux, e.g. `scripts/launch.sh grav_comp real` or `scripts/launch.sh ik_qp real` (with `run_scene_info.py`).

---

## 4. Things that differ from the simulation

- **Friction.** The friction feed-forward (`Fjc` in [06](06_inverse_kinematics_quadratic_program.md) / [07](07_computed_torque_control.md)) uses the MuJoCo values. Re-identify them on your arm (e.g. slow constant-velocity moves per joint, read the torque needed).
- **Torque sensing.** `tau` in `franka_arm_joint_meas` is the robot’s `tau_J`. Mass estimation ([12](12_force_aware_grasping.md)) works the same way; keep the tare step (sensor offsets drift).
- **Object pose for grasping.** In simulation the env publishes `franka_object_pose`. On the robot it comes from perception (section 6).
- **Timing.** The workstation should be dedicated to the robot while controlling it. The grasping controllers fall back to holding if their planner stalls (`max_plan_age`), but the arm then stops mid-motion.

---

## 5. How it is implemented

| Piece | Role |
|-------|------|
| `run_hardware.py` | Builds the platform + LCM managers; always shuts torque control down on exit |
| `robot_platform/hardware/franka_fr3/HardwareFrankaFR3.py` | Command law, safety layer, gripper threads; `PandaPyBackend` / `FakeBackend` |
| `configs/franka_fr3/hardware/fr3.yaml` | Hostname, safety limits, channels |
| `run_scene_info.py`, `configs/franka_fr3/hardware/scene_info.yaml` | Collision geometry of the real cell |
| `run_pose_bridge.py`, `configs/franka_fr3/perception/pose_bridge.yaml` | FoundationPose pose -> `franka_object_pose` |
| `scripts/preflight.py`, `scripts/launch.sh` | Workstation check, tmux launcher |
| `tests/unit/test_hardware_safety.py`, `tests/integration/test_real_pipeline.py` | Command law and safety layer; fake-robot pipeline |

---

## 6. Grasping from perception

```
run_sensors.py (RealSense)  --rgbd d455_1-->  run_perception.py (FoundationPose + SAM2, GPU)
      --fp_obj_pose-->  run_pose_bridge.py  --franka_object_pose-->  grasp controller (--env_mode real)
```

FoundationPose publishes the pose of the object's *oriented bounding box* in the camera's "world" frame. The grasp controllers need the *mesh frame* in the *robot base* frame. `run_pose_bridge.py` converts:

$$
T_{\mathrm{base,mesh}} = T_{\mathrm{base,world}}\; T_{\mathrm{world,bb}}\; T_{\mathrm{to\_origin}},
$$

where `to_origin` comes from `trimesh.bounds.oriented_bounds` of the same mesh FoundationPose loads. If the camera extrinsics are calibrated against the robot base ([14](14_diff_camera_calibration.md)), world = base and `base_from_world` stays identity.

Before the first real grasp:

1. **Calibrate the camera** ([14](14_diff_camera_calibration.md)) and put the extrinsics in `configs/sensors/realsense/one_realsense.yaml`. The values there are from an older setup.
2. **Check the pose** without the arm moving: run the sensors, perception and the bridge, plus the visualizer, and compare the object with reality. Move the object, and the pose should follow.
3. **Describe the table** in the scene used by `run_scene_info.py`, so the arm keeps clear of it.
4. Run it (the configs are for the YCB mustard bottle; change `object_config`, `fp_mesh_file` and the FoundationPose `mesh_file` together):
   ```bash
   scripts/launch.sh grasp_perception real        # or force_grasp_perception
   ```
   Press `v` in the controller pane. Keep the user stop in hand.

In simulation the same chain runs with a simulated camera: `scripts/launch.sh grasp_perception` (env `grasping_perception.yaml`: the env publishes the camera but no ground-truth object pose).
