from queue import Queue

from communication.lcm.subscriber.BaseSubManager import BaseControllerSubManager
from communication.lcm.subscriber.data_subscriber.ClockSubscriber import ClockSubscriber
from communication.lcm.subscriber.data_subscriber.ColInfoSubscriber import (
    ColInfoSubscriber,
)
from communication.lcm.subscriber.data_subscriber.GripperStateSubscriber import (
    GripperStateSubscriber,
)
from communication.lcm.subscriber.data_subscriber.ImpGainsCmdSubscriber import (
    ImpGainsCmdSubscriber,
)
from communication.lcm.subscriber.data_subscriber.JointMeasSubscriber import (
    JointMeasSubscriber,
)
from communication.lcm.subscriber.data_subscriber.NamedVecListSubscriber import (
    NamedVecListSubscriber,
)
from communication.lcm.subscriber.data_subscriber.SE3PoseSubscriber import (
    SE3PoseSubscriber,
)
from communication.lcm.subscriber.data_subscriber.P2PGainsCmdSubscriber import (
    P2PGainsCmdSubscriber,
)


class FrankaSubManager(BaseControllerSubManager):
    """Controller-side subscriber for the FR3 + Franka Hand.

    ``intr_sub_que_dict`` gets one queue per channel: arm joint_meas (7),
    finger joint_meas (2), and the optional ones below.

    Optional channels (empty string = not subscribed):
      * ``gripper_state_channel``   gripper_state_t            -> intr
      * ``gains_cmd_channel``       Viser gain sliders         -> intr
        (``gains_cmd_type``: "imp" = imp_gains_cmd_t for the impedance
        controller, "p2p" = p2p_gains_cmd_t for min-jerk P2P)
      * ``robot_col_info_channel``  col_info_t, link-local     -> intr
      * ``static_col_info_channel`` col_info_t, world-fixed    -> extr
      * ``object_pose_channel``     se3_pose_t (object)        -> extr
      * ``object_physics_channel``  vec_list_t (mass, COM, g)  -> extr
    """

    def __init__(
        self,
        arm_joint_meas_channel: str = "franka_arm_joint_meas",
        gripper_joint_meas_channel: str = "franka_gripper_joint_meas",
        gripper_state_channel: str = "",
        gains_cmd_channel: str = "",
        gains_cmd_type: str = "imp",
        robot_col_info_channel: str = "",
        static_col_info_channel: str = "",
        object_pose_channel: str = "",
        object_physics_channel: str = "",
        num_arm_joints: int = 7,
        num_finger_joints: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        intr = self.data_queue_dict["intr_sub_que_dict"]

        for ch, n in (
            (arm_joint_meas_channel, num_arm_joints),
            (gripper_joint_meas_channel, num_finger_joints),
        ):
            # Latest wins: controllers read one sample per loop iteration, so
            # an unbounded FIFO would build up stale state (seconds of lag).
            intr[ch] = Queue(maxsize=1)
            JointMeasSubscriber(self._lcm_instance, intr[ch], num_joints=n).subscribe(
                ch
            )

        if gripper_state_channel:
            intr[gripper_state_channel] = Queue()
            GripperStateSubscriber(
                self._lcm_instance, intr[gripper_state_channel]
            ).subscribe(gripper_state_channel)

        if gains_cmd_channel:
            sub_cls = {"imp": ImpGainsCmdSubscriber, "p2p": P2PGainsCmdSubscriber}[
                gains_cmd_type
            ]
            intr[gains_cmd_channel] = Queue()
            sub_cls(self._lcm_instance, intr[gains_cmd_channel]).subscribe(
                gains_cmd_channel
            )

        if robot_col_info_channel:
            intr[robot_col_info_channel] = Queue()
            ColInfoSubscriber(
                self._lcm_instance, intr[robot_col_info_channel]
            ).subscribe(robot_col_info_channel)

        if static_col_info_channel:
            extr = self.data_queue_dict["extr_sub_que_dict"]
            extr[static_col_info_channel] = Queue()
            ColInfoSubscriber(
                self._lcm_instance, extr[static_col_info_channel]
            ).subscribe(static_col_info_channel)

        extr = self.data_queue_dict["extr_sub_que_dict"]
        for ch, cls in (
            (object_pose_channel, SE3PoseSubscriber),
            (object_physics_channel, NamedVecListSubscriber),
        ):
            if ch:
                extr[ch] = Queue()
                cls(self._lcm_instance, extr[ch]).subscribe(ch)

        ClockSubscriber(
            self._lcm_instance, self.data_queue_dict["time_sub_que"]
        ).subscribe("sim_clock")
