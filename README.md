# Franka FR3 tutorials: control, grasping, perception

Torque-level control of a **Franka Research 3** with a **Robotiq 2F-85**
gripper and **ZED** cameras (the DROID setup), in MuJoCo and on the real
robot, all in Python. Each piece runs as its own process:
the simulator or the robot, a controller, a visualizer, and perception. They
talk over **LCM** (there is no ROS in this repo).

```
            joint_ctrl_t (q_des, qd_des, tau_ff, kp, kd) @ 1 kHz
controller  ─────────────────────────────────────────────►  run_env.py      (MuJoCo, SimFrankaFR3)
            ◄─────────────────────────────────────────────    or
            joint_meas_t (q, qd, tau_J), gripper_state_t      run_hardware.py (panda-py → FR3)
                  ▲
                  │ franka_object_pose (se3_pose_t)
run_pose_bridge.py ◄── fp_obj_pose ── run_perception.py (FoundationPose) ◄── RGB-D camera (ZED)
```

Both the simulator and the robot apply the same law to every command:
`tau = kp (q_des − q) + kd (qd_des − qd) + tau_ff`. Gravity is compensated on
the plant side, as the FR3 firmware does. A controller therefore runs
unchanged on either one; only `--env_mode real` changes for the robot.

**Start here:** [STUDY_GUIDE.md](STUDY_GUIDE.md), an ordered path through the
code (what to read, then what to run).

## Setup

```bash
conda activate franka                 # python 3.12; see requirements.txt
pip install -r requirements.txt       # if setting up a new machine
```

- Headless machine (no display): add `--vis_mode vis_off` to `run_env.py`, and
  set `MUJOCO_GL=osmesa` for camera envs.
- Real robot (workstation only): you need a PREEMPT_RT kernel and panda-py
  built for your FR3's libfranka. Check the workstation with
  `python scripts/preflight.py` and see [tutorial 15](tutorial/15_real_fr3.md).
- Perception (tutorials 13, 14): needs a CUDA GPU (FoundationPose, SAM2,
  nvdiffrast).

## Quick start (simulation)

Each process goes in its own terminal:

```bash
python run_env.py        --config configs/franka_fr3/env/default.yaml
python run_controller.py --config configs/franka_fr3/controllers/grav_comp.yaml
python run_visualizer.py --config configs/visualizer/franka_fr3/default.yaml   # http://localhost:8080
```

Or start them all in tmux:

```bash
scripts/launch.sh list                 # targets
scripts/launch.sh ik_qp                # tutorial 06 in the sim
HEADLESS=1 scripts/launch.sh grasp     # tutorial 10, no MuJoCo window
scripts/launch.sh ik_qp real           # same controller on the FR3
```

Any YAML value can be overridden on the command line, e.g.
`--platform.gravity_comp false` or `--env_mode real`.

## Tutorials

| # | Topic | Controller / script |
|---|-------|---------------------|
| 01–02 | Quick start, the MuJoCo env | `grav_comp` |
| 03 | Gravity compensation (plant vs controller) | `GravCompControl` |
| 04–05 | Collision geometry, collision-aware float | `CollisionAwareGravCompControl` |
| 06 | Cartesian IK as a QP (velocity control) | `IKQPControl` |
| 07 | RRT + minimum-jerk + computed torque | `MinimumJerkP2PControl` |
| 08 | Joint / task-space impedance | `ImpedanceControl` |
| 09 | IK to a sampled pose + joint impedance | `IKReachControl` |
| 10 | Reactive parallel-jaw grasping | `ParallelJawGraspControl` |
| 11 | Grasp quality: grip-force model vs sim | `run_grasp_quality.py` |
| 12 | Force-aware grasping (weigh, re-grasp) | `ForceAwareGraspControl` |
| 13 | FoundationPose on a simulated camera | `run_perception.py` |
| 14 | Camera calibration by differentiable rendering | `run_camera_calibration.py` |
| 15 | Running on the real FR3 | `run_hardware.py` |

## Repository map

| Path | What |
|------|------|
| `run_*.py` | Process entry points (env, hardware, controller, visualizer, sensors, perception, pose bridge, scene info, grasp quality, calibration) |
| `controller/` | Controllers; `BaseController` has the control loop |
| `env/` | MuJoCo envs (`BaseEnv` loop, collision / grasp / camera variants) |
| `robot_platform/sim/franka_fr3/` | Sim plant: command law, gravity comp, gripper emulation, tau_J |
| `robot_platform/hardware/franka_fr3/` | Real plant: panda-py backend, safety layer, Robotiq driver (`robotiq.py`, Modbus over USB) |
| `robot_platform/gripper/robotiq_2f85.py` | 2F-85 geometry and ranges shared by sim, hardware, visualizer, grasping |
| `communication/lcm/` | Pub/sub managers (one per process type) and per-message publishers/subscribers |
| `data_type/`, `lcm_type/` | In-process data classes and LCM message types |
| `utils/` | Pinocchio / FCL helpers, grasping (`utils/grasping`), collision, calibration |
| `perception/`, `sensor/` | FoundationPose + SAM2; ZED (DROID) and RealSense camera drivers |
| `visualizer/` | viser-based 3D visualizer |
| `configs/` | All YAML configs (`configs/franka_fr3/*` for the robot) |
| `assets/scene/franka_fr3/` | FR3 + Robotiq 2F-85 MJCF, scenes, URDFs (articulated for Viser, rigid gripper for the controllers) |
| `tests/` | Unit + integration tests |
| `scripts/` | `launch.sh` (tmux), `preflight.py` (robot workstation), `lcm_echo.py` (watch the LCM traffic) |

## Tests

```bash
pytest -m "not integration"     # ~15 s: models, collision model, gripper, safety layer, grasp math, configs, imports
pytest -m integration           # ~10 min: launches env/controller/hardware processes headless
pytest                          # everything
```

Integration tests run on their own LCM port, so they don't interfere with a
sim or robot that is already running. They cover tutorials 01–12, the fake
hardware backend with a `--env_mode real` controller, the pose bridge and the
camera env. The GPU parts (FoundationPose, SAM2, differentiable rendering) are
not tested.

Integration tests run in real time. On an overloaded shared machine (the pytest
header shows the load) they can time out. Rerun a failing test on its own
before suspecting the code. Each test's process logs are kept under
`/tmp/pytest-of-$USER/pytest-current/<test>/`.
