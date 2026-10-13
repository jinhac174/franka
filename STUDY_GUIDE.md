# Study guide

The plan for studying the whole repo, one file at a time, before deploying on
the real FR3. The main thread is **how robot control is implemented**: one
command's path from a controller's computation to a torque on a joint, in
simulation and on the robot.

**How each session works**

1. Reading map: what the file is for, what each block of lines does, what matters, what to skip.
2. Read the file with the map next to it; collect questions.
3. Answers, then a small experiment (run something, change a value, watch it).
4. Checkpoint questions. If you can answer them, move on.

There is no ROS in this repo: processes talk over LCM.

---

## Part A: the core

| ✓ | # | Module | Files | You'll understand |
|---|---|---|---|---|
| ☐ | 0 | Setup and viewing | (none) | SSH tunnel, the browser view (Viser), tmux, `scripts/lcm_echo.py` |
| ☐ | 1 | Messages | `lcm_type/lcm_type/{joint_meas_data,joint_ctrl_data,gripper_cmd,gripper_state,clock}.lcm`, `lcm_type/joint/joint_meas_t.py`, `data_type/BaseData.py`, `data_type/basic_types/{JointMeasData,JointCtrlData}.py` | What travels between programs; wire format vs in-program objects |
| ☐ | 2 | Communication layer | `communication/lcm/subscriber/{BaseDataSubscriber,BaseSubManager}.py`, `.../data_subscriber/JointMeasSubscriber.py`, `.../sub_manager/controller/FrankaSubManager.py`, `communication/lcm/publisher/{BaseDataPublisher,BasePubManager}.py`, `.../data_publisher/JointCtrlPublisher.py`, `.../pub_manager/controller/JointCtrlPubManager.py` | Threads, queues, latest wins, YAML channel → subscription |
| ☐ | 3 | Robot model files | `assets/scene/franka_fr3/fr3_robotiq_robot.xml`, `scene_default.xml`, `urdf/fr3_robotiq.urdf` + `fr3_robotiq_rigid.urdf` (skim), `robot_platform/gripper/robotiq_2f85.py` | MuJoCo XML (bodies, joints, actuators, geoms, the 2F-85 linkage); why URDFs too (Pinocchio, Viser) |
| ☐ | 4 | Simulator side | `run_env.py`, `env/__init__.py`, `env/BaseEnv.py`, `env/MujocoBaseEnv.py`, `robot_platform/sim/franka_fr3/SimFrankaFR3.py`, env sub/pub managers, `configs/franka_fr3/env/default.yaml` | Physics step, time modes, command law, plant gravity comp, simulated gripper, measured torque |
| ☐ | 5 | Controller side | `run_controller.py`, `controller/__init__.py`, `controller/BaseController.py`, `controller/GravCompControl.py`, `configs/franka_fr3/controllers/grav_comp.yaml`; tutorials 01–03 | The 1 kHz loop, time modes, keys, the simplest controller end to end |
| ☐ | 6 | Visualizer (Viser) | `run_visualizer.py`, `visualizer/BaseVisManager.py`, `visualizer/vis_manager/franka_fr3/FrankaVisManager.py`, `visualizer/data_visualizer/RobotArmVisualizer.py` (+ Mesh/Pose visualizers) | How joint angles become a robot in your browser; GUI elements |

## Part B: one tutorial at a time

| ✓ | # | Tutorials | Main files | New idea |
|---|---|---|---|---|
| ☐ | 7 | 04–05 Collision | `utils/mujoco/collision.py`, `env/MujocoColEnv.py`, `run_scene_info.py`, `controller/CollisionAwareGravCompControl.py` | Robot and world as primitives; repulsion from obstacles |
| ☐ | 8 | 06 IK as a QP | `controller/IKQPControl.py`, `utils/pinocchio/` | Cartesian motion via an optimizer |
| ☐ | 9 | 07 Planning + computed torque | `controller/MinimumJerkP2PControl.py` | RRT, min-jerk trajectories, dynamics model |
| ☐ | 10 | 08–09 Impedance, IK reach | `controller/ImpedanceControl.py`, `controller/IKReachControl.py` | Joint / task-space springs |
| ☐ | 11 | 10–12 Grasping | `utils/grasping/`, `env/MujocoGraspEnv.py`, `controller/ParallelJawGraspControl.py`, `run_grasp_quality.py`, `controller/ForceAwareGraspControl.py` | Antipodal grasps, approach/lift, grip force, weighing |
| ☐ | 12 | 13–14 Perception (read only here: no GPU) | `env/MujocoCamEnv.py`, `run_perception.py`, `perception/foundation_pose/`, `run_pose_bridge.py`, `run_camera_calibration.py` | Camera → object pose → grasp |

## Part C: the real robot

| ✓ | # | Module | Files | New idea |
|---|---|---|---|---|
| ☐ | 13 | Hardware side | `run_hardware.py`, `robot_platform/hardware/franka_fr3/{HardwareFrankaFR3,robotiq}.py`, `configs/franka_fr3/hardware/fr3.yaml`, `sensor/sensor/zed/ZED.py`, `configs/sensors/zed/droid.yaml`, tutorial 15 | Same job as `SimFrankaFR3`, on the robot; safety layer; panda-py / libfranka; the Robotiq over Modbus; ZED cameras |
| ☐ | 14 | Tests and deployment | `tests/`, `scripts/preflight.py`, `scripts/launch.sh` | How each layer is checked; bring-up order |

---

## Viewing on this headless server

Each program reads keys from its own terminal, so give each one its own tmux
pane. In every pane first run `conda activate franka && cd ~/projects/franka`.

```bash
python run_env.py        --config configs/franka_fr3/env/default.yaml --vis_mode vis_off   # pane 1: simulator
python run_controller.py --config configs/franka_fr3/controllers/grav_comp.yaml            # pane 2: controller
python run_visualizer.py --config configs/visualizer/franka_fr3/default.yaml               # pane 3: browser view
python scripts/lcm_echo.py                                                                 # pane 4: watch traffic
```

Press `q` in a pane to stop that program. Pane 3 prints the port it got
(add `--port 8123` to pick one). On your **laptop** (not the server), forward
it and open `http://localhost:8123`:

```bash
ssh -N -L 8123:127.0.0.1:8123 danielc174@165.132.142.208
```

---

## Deployment (Part C)

### On the workstation, once

```bash
python scripts/preflight.py                         # kernel, rtprio, panda-py, LCM, config
python scripts/preflight.py --connect               # + panda-py connects, reads q (no motion)
pytest -m "not integration"                          # same code, new machine
pytest tests/integration/test_real_pipeline.py      # fake backend, scene info, bridge
```
Set `robot.hostname` in `configs/franka_fr3/hardware/fr3.yaml`. Describe your
cell (table, walls) in a scene XML for `run_scene_info.py`; the default
scene's walls are placeholders.

### First session (user stop in hand, one step at a time)

| Step | Command | Expect |
|---|---|---|
| 1 | `run_hardware.py ... --robot.backend fake` + `grav_comp --env_mode real` | Dry run; no robot |
| 2 | `run_hardware.py --config configs/franka_fr3/hardware/fr3.yaml` alone | Arm holds (timeout hold) |
| 3 | `scripts/launch.sh grav_comp real` | Arm floats under your hand; check the gripper load in Desk (0.925 kg) if it drifts |
| 4 | `run_controller.py --config configs/franka_fr3/controllers/impedance.yaml --env_mode real --kq_scale 0.3` (with hardware + visualizer) | Springs back to the start pose; then full gains |
| 5 | `scripts/launch.sh col_aware real` / `ik_qp real` | Needs your cell in `scene_info` |
| 6 | `min_jerk real`, `ik_reach real` | Planned motions; start with small ones |
| 7 | `scripts/launch.sh grasp_perception real` | Camera → FoundationPose → bridge → grasp |

### The full real grasping pipeline

```
run_sensors.py (ZED ext1, droid.yaml) ─rgbd─► run_perception.py (foundation_pose_real.yaml) ─fp_obj_pose─►
run_pose_bridge.py ─franka_object_pose─► run_controller.py (parallel_jaw_grasp / force_aware_grasp --env_mode real)
run_scene_info.py ─franka_static_col_info / franka_robot_col_info─► controller
run_hardware.py ◄─joint_ctrl / gripper_cmd─► controller
```

Known differences from the sim (tutorial 15 §4): friction values, real
`tau_J` offsets (keep the tare step), camera extrinsics, and timing (keep the
workstation dedicated to the robot).

---

## Appendix: Running the tests

| File | What it proves |
|---|---|
| `tests/unit/test_models.py` | MJCF and both URDFs agree (TCP, gripper linkage, gravity); 2F-85 tables and mass; plant gravcomp; tau_J = gravity load |
| `tests/unit/test_collision_model.py` | Capsules cover the link meshes; config pairs name real geoms |
| `tests/unit/test_gripper_sim.py` | Gripper API semantics (move, grasp, fail, seq, stop, homing) |
| `tests/unit/test_hardware_safety.py` | Command law + every safety mechanism + gripper round trip |
| `tests/unit/test_robotiq.py` | Robotiq Modbus frames (vs the manual) and move / grasp / stop / homing against a simulated gripper |
| `tests/unit/test_grasping_math.py` | Antipodal sampling, TCP poses (pad arc), gripper boxes cover the 2F-85, grip force, pose bridge |
| `tests/unit/test_configs_and_imports.py` | Every YAML loads; every path / command in the docs exists; every module imports |
| `tests/integration/test_sim_basic.py` | 01–05, 08: rates, gripper over LCM, collision info, impedance modes |
| `tests/integration/test_sim_motion.py` | 06, 07, 09: IK-QP nudges, min-jerk to goal, IK reach to pose |
| `tests/integration/test_sim_grasping.py` | 10–12: grasp and lift, model vs sim, weigh and re-grasp |
| `tests/integration/test_real_pipeline.py` | Fake hardware + scene info + real-mode controller, timeout hold, pose bridge, camera env |

`tests/helpers.py` (`Procs`, `Listener`) is a handy pattern for scripting
experiments of your own: start processes, send keys, listen on LCM, measure.
