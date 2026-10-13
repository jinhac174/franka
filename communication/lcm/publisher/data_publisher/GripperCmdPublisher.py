from queue import Queue

import lcm

from communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from data_type.basic_types.GripperCmdData import GripperCmdData
from lcm_type.gripper.gripper_cmd_t import gripper_cmd_t


class GripperCmdPublisher(BaseDataPublisher):
    """Publish GripperCmdData as gripper_cmd_t."""

    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        if not self._data_que.empty():
            data = self._data_que.get()
            if isinstance(data, GripperCmdData):
                t, seq, mode, width, speed, force, eps_in, eps_out = data.get_data()
                msg = gripper_cmd_t()
                msg.timestamp = t
                msg.seq = seq
                msg.mode = mode
                msg.width = width
                msg.speed = speed
                msg.force = force
                msg.epsilon_inner = eps_in
                msg.epsilon_outer = eps_out
                self._lcm_instance.publish(self._channel, msg.encode())
