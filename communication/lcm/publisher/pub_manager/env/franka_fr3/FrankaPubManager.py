from queue import Queue

from communication.lcm.publisher.BasePubManager import BaseEnvPubManager
from communication.lcm.publisher.data_publisher.CameraInfoPublisher import (
    CameraInfoPublisher,
)
from communication.lcm.publisher.data_publisher.ColInfoPublisher import (
    ColInfoPublisher,
)
from communication.lcm.publisher.data_publisher.GripperStatePublisher import (
    GripperStatePublisher,
)
from communication.lcm.publisher.data_publisher.JointMeasPublisher import (
    JointMeasPublisher,
)
from communication.lcm.publisher.data_publisher.NamedVecListPublisher import (
    NamedVecListPublisher,
)
from communication.lcm.publisher.data_publisher.RGBDPublisher import RGBDPublisher
from communication.lcm.publisher.data_publisher.SE3PosePublisher import (
    SE3PosePublisher,
)


class FrankaPubManager(BaseEnvPubManager):
    """Env-side publisher for the FR3 + Franka Hand.

    Channel names must match the ones the SimFrankaFR3 platform writes to
    (the env config passes the same keys to both).

    Optional collision info (for MujocoColEnv): ``robot_col_info_channel``
    (intr, link-local primitives) and ``static_col_info_channel`` (extr,
    world-fixed scene geoms).

    Optional object info (for MujocoGraspEnv, extr): ``object_pose_channel``
    (se3_pose_t) and ``object_physics_channel`` (vec_list_t).

    Optional cameras (for MujocoCamEnv, extr): ``camera_channels``, a list of
    channel names; each gets an RGB-D publisher and ``<channel>_info``.
    """

    def __init__(
        self,
        *args,
        arm_joint_meas_channel: str = "franka_arm_joint_meas",
        gripper_joint_meas_channel: str = "franka_gripper_joint_meas",
        gripper_state_channel: str = "franka_gripper_state",
        robot_col_info_channel: str = "",
        static_col_info_channel: str = "",
        object_pose_channel: str = "",
        object_physics_channel: str = "",
        camera_channels=(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        intr = self.pub_que_dict["intr_pub_que_dict"]
        for ch in (arm_joint_meas_channel, gripper_joint_meas_channel):
            intr[ch] = Queue()
            self.publisher_dict[ch] = JointMeasPublisher(
                self._lcm_instance, ch, intr[ch]
            )
        if gripper_state_channel:
            intr[gripper_state_channel] = Queue()
            self.publisher_dict[gripper_state_channel] = GripperStatePublisher(
                self._lcm_instance, gripper_state_channel, intr[gripper_state_channel]
            )
        if robot_col_info_channel:
            intr[robot_col_info_channel] = Queue()
            self.publisher_dict[robot_col_info_channel] = ColInfoPublisher(
                self._lcm_instance, robot_col_info_channel, intr[robot_col_info_channel]
            )
        if static_col_info_channel:
            extr = self.pub_que_dict["extr_pub_que_dict"]
            extr[static_col_info_channel] = Queue()
            self.publisher_dict[static_col_info_channel] = ColInfoPublisher(
                self._lcm_instance,
                static_col_info_channel,
                extr[static_col_info_channel],
            )
        extr = self.pub_que_dict["extr_pub_que_dict"]
        for ch, cls in (
            (object_pose_channel, SE3PosePublisher),
            (object_physics_channel, NamedVecListPublisher),
        ):
            if ch:
                extr[ch] = Queue()
                self.publisher_dict[ch] = cls(self._lcm_instance, ch, extr[ch])
        for ch in camera_channels or ():
            for name, cls in ((ch, RGBDPublisher), (ch + "_info", CameraInfoPublisher)):
                extr[name] = Queue()
                self.publisher_dict[name] = cls(self._lcm_instance, name, extr[name])
