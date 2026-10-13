from queue import Queue

import lcm

from communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from data_type.basic_types.GripperCmdData import GripperCmdData
from lcm_type.gripper.gripper_cmd_t import gripper_cmd_t


class GripperCmdSubscriber(BaseDataSubscriber):
    """Decode gripper_cmd_t into GripperCmdData."""

    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        msg = gripper_cmd_t.decode(data)
        cmd = GripperCmdData()
        cmd.set_data(
            msg.timestamp,
            msg.seq,
            msg.mode,
            msg.width,
            msg.speed,
            msg.force,
            msg.epsilon_inner,
            msg.epsilon_outer,
        )
        self.data_queue.put(cmd)
