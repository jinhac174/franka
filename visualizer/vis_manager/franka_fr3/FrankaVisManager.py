from queue import Queue

from omegaconf import DictConfig

from visualizer.BaseVisManager import BaseVisManager
from visualizer.data_visualizer.RobotArmVisualizer import (
    RobotArmVisualizer,
)


class FrankaVisManager(BaseVisManager):
    """Viser visualizer for the Franka FR3 + Robotiq 2F-85.

    Subscribes to the arm (7) and gripper (1: driver angle) joint_meas_t
    channels and renders urdf/fr3_robotiq.urdf ([arm(7), driver]; the rest of
    the gripper linkage follows via <mimic>). Sim and hardware publish the
    same channels, so the config works for both.
    """

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, max_sub_freq=1000, *args, **kwargs)

    def _set_visualizers(self, config: DictConfig):
        fcfg = config["franka_fr3"]
        arm_channel = fcfg["arm_joint_meas_channel"]
        gripper_channel = fcfg["gripper_joint_meas_channel"]

        intr = self.data_queue_dict["intr_sub_que_dict"]
        intr[arm_channel] = Queue()
        intr[gripper_channel] = Queue()

        # Remaining keys (collision / target / trajectory overlays, ...) are
        # passed straight through to the visualizer.
        extra = {
            k: v
            for k, v in fcfg.items()
            if k
            not in (
                "arm_joint_meas_channel",
                "gripper_joint_meas_channel",
                "num_arm_joints",
                "num_gripper_joints",
            )
        }
        extra.setdefault("traj_frame", "fr3_gripper_tcp")

        vis = RobotArmVisualizer(
            self.viser,
            self._lcm_instance,
            intr[arm_channel],
            intr[gripper_channel],
            arm_joint_meas_channel=arm_channel,
            hand_joint_meas_channel=gripper_channel,
            num_arm_joints=fcfg.get("num_arm_joints", 7),
            num_hand_joints=fcfg.get("num_gripper_joints", 2),
            **extra,
        )

        self._list_visualizers = [vis]
        print(
            f"[Visualizer] FrankaVisManager initialized "
            f"(sub freq={self._max_sub_freq} Hz, vis freq={self._vis_freq} Hz)."
        )
