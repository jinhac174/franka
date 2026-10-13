from queue import Queue

from communication.lcm.publisher.BasePubManager import BaseCtrlPubManager
from communication.lcm.publisher.data_publisher.JointCtrlPublisher import (
    JointCtrlPublisher,
)
from communication.lcm.publisher.data_publisher.NamedVecListPublisher import (
    NamedVecListPublisher,
)
from communication.lcm.publisher.data_publisher.Pose3DPublisher import Pose3DPublisher
from communication.lcm.publisher.data_publisher.SE3PosePublisher import (
    SE3PosePublisher,
)


class JointCtrlGraspPubManager(BaseCtrlPubManager):
    """Joint ctrl (+ gripper, via the base class) and grasp overlays for Viser:

    * ``pose_channel``             pose3d_t   IK target / TCP (keyboard mode)
    * ``target_tcp_channel``       se3_pose_t current goal TCP pose
    * ``grasp_candidates_channel`` vec_list_t selected contact pair (c0_a/c0_b)
    """

    def __init__(
        self,
        ctrl_channel: str = "franka_arm_joint_ctrl",
        pose_channel: str = "",
        target_tcp_channel: str = "",
        grasp_candidates_channel: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.publisher_dict[ctrl_channel] = JointCtrlPublisher(
            self._lcm_instance, ctrl_channel, self.ctrl_pub_que
        )
        for ch, cls in (
            (pose_channel, Pose3DPublisher),
            (target_tcp_channel, SE3PosePublisher),
            (grasp_candidates_channel, NamedVecListPublisher),
        ):
            if ch:
                self.pub_que_dict[ch] = Queue(maxsize=1)
                self.publisher_dict[ch] = cls(self._lcm_instance, ch, self.pub_que_dict[ch])
