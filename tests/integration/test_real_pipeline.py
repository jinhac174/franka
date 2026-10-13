"""The real-robot pipeline, everything except the robot and the GPU:

* run_hardware.py (fake backend) + run_scene_info.py + a collision-aware
  controller with --env_mode real (tutorial 15),
* perception -> grasp bridge: run_pose_bridge.py fed by the perception
  tutorial's ground-truth bounding-box stream (same format as FoundationPose),
* the grasping-with-camera env (grasping_perception.yaml) publishes the
  camera, not the object pose (that must come from perception).
"""

import time

import lcm
import numpy as np

from data_type.basic_types.GripperCmdData import GripperCmdData
from lcm_type.gripper.gripper_cmd_t import gripper_cmd_t
from lcm_type.vision.rgbd_t import rgbd_t
from tests.helpers import STARTUP, CTRL, HOME_Q, Listener, arm_q, decoders

HW = "run_hardware.py --config configs/franka_fr3/hardware/fr3.yaml --robot.backend fake"


def test_fake_hardware_scene_info_and_real_mode_controller(procs, lcm_listener):
    lst = lcm_listener
    procs.start("hw", HW)
    assert lst.wait_for(lambda: "franka_arm_joint_meas" in lst.last, STARTUP), "hardware did not publish"
    procs.start("scene", "run_scene_info.py --config configs/franka_fr3/hardware/scene_info.yaml")
    assert lst.wait_for(lambda: "franka_static_col_info" in lst.last
                        and "franka_robot_col_info" in lst.last, STARTUP), "no scene info"
    q0 = arm_q(lst)
    assert np.max(np.abs(q0 - HOME_Q)) < 0.02  # fake robot starts at home, holds

    # The collision-aware controller needs the col info (it waits for it).
    procs.start("ctrl", CTRL, "configs/franka_fr3/controllers/col_aware_grav_comp.yaml",
                "--env_mode real")
    assert lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, STARTUP), "no controller"
    lst.reset_counts()
    lst.spin(2.0)
    rate = lst.count["franka_arm_joint_meas"] / 2.0
    assert rate > 500, rate  # nominal 1 kHz
    assert np.max(np.abs(arm_q(lst) - q0)) < 0.05

    # Gripper through the hardware process (fake hand).
    cmd = gripper_cmd_t()
    cmd.timestamp, cmd.seq, cmd.mode = time.time(), 3, GripperCmdData.MOVE
    cmd.width, cmd.speed, cmd.epsilon_inner, cmd.epsilon_outer = 0.05, 0.2, 0.005, 0.005
    lcm.LCM().publish("franka_gripper_cmd", cmd.encode())
    gs = lambda: lst.last.get("franka_gripper_state")  # noqa: E731
    assert lst.wait_for(lambda: gs() is not None and gs().last_seq == 3 and not gs().busy, 10)
    assert gs().last_success and abs(gs().width - 0.05) < 1e-6


def test_hardware_holds_when_controller_stops(procs, lcm_listener):
    """Command timeout: the controller dies -> the hardware holds the pose."""
    lst = lcm_listener
    procs.start("hw", HW)
    assert lst.wait_for(lambda: "franka_arm_joint_meas" in lst.last, STARTUP)
    ctrl = procs.start("ctrl", CTRL, "configs/franka_fr3/controllers/grav_comp.yaml", "--env_mode real")
    assert lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, STARTUP)
    ctrl.stop()
    lst.spin(2.0)
    m = lst.last["franka_arm_joint_meas"]
    assert np.max(np.abs(np.asarray(m.qd))) < 0.01
    assert np.max(np.abs(arm_q(lst) - HOME_Q)) < 0.05


def test_pose_bridge_turns_bb_pose_into_object_pose(procs, lcm_listener):
    """Bounding-box pose in the camera world frame -> mustard bottle mesh frame
    in the base frame: upright, base on the 0.42 m table at (0, 0)."""
    lst = lcm_listener
    procs.start("env", "run_env.py --vis_mode vis_off --config "
                "configs/perception_tutorial/env/table_mesh_cam.yaml")
    procs.start("bridge", "run_pose_bridge.py --config configs/franka_fr3/perception/pose_bridge.yaml",
                "--fp_pose_channel sim_obj_pose_bb2world")
    assert lst.wait_for(lambda: lst.count["franka_object_pose"] > 10, STARTUP), "bridge did not publish"
    msg = lst.last["franka_object_pose"]
    p = np.asarray(msg.position)
    w, x, y, z = msg.quat_wxyz
    R22 = 1 - 2 * (x * x + y * y)  # z-axis of the object . world z
    assert np.allclose(p, [0.0, 0.0, 0.42], atol=0.01), p
    assert R22 > 0.99


def test_grasping_perception_env_publishes_camera_not_object_pose(procs):
    lst = Listener({**decoders(), "scene_camera": rgbd_t.decode})
    # Small images: a 640x480 RGBD frame (~2.4 MB) exceeds a default kernel
    # UDP receive buffer (rmem_max 208 KB) and is dropped; tutorial 13 shows
    # how to raise it for the real thing.
    procs.start("env", "run_env.py --vis_mode vis_off --config "
                "configs/franka_fr3/env/grasping_perception.yaml",
                "--sensor.extrinsic.scene_camera.width 160 --sensor.extrinsic.scene_camera.height 120")
    assert lst.wait_for(lambda: lst.count["scene_camera"] >= 3, STARTUP), "no camera images"
    img = lst.last["scene_camera"]
    assert (img.width, img.height) == (160, 120) and img.rgb_size > 0 and img.depth_size > 0
    depth = np.frombuffer(bytes(img.depth_image), dtype=np.float32)
    assert np.isfinite(depth).all() and 0.2 < np.median(depth) < 3.0  # sees the scene
    lst.spin(2.0)
    assert lst.count["franka_object_pose"] == 0  # must come from perception
    assert lst.count["franka_arm_joint_meas"] > 0
