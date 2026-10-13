"""Run a real robot (hardware platform) with LCM pub/sub managers.

It replaces run_env.py on the robot: same channels, real measurements.
"""

import argparse

from omegaconf import DictConfig, OmegaConf

from communication.lcm.publisher.pub_manager.hardware import get_hardware_pub_manager
from communication.lcm.subscriber.sub_manager.hardware import get_hardware_sub_manager
from robot_platform.hardware import get_hardware_platform
from utils.terminal.argparse_utils import parse_nested_args, parse_unknown_args


def run_hardware(cfg_platform: DictConfig) -> None:
    """Create the platform + LCM managers, run until quit, always shut down."""
    hardware_platform = get_hardware_platform(cfg_platform)

    sub = get_hardware_sub_manager(cfg_platform["sub_manager"])
    hardware_platform.set_sub_que_dict(sub.get_sub_que_dict())

    pub = get_hardware_pub_manager(cfg_platform["pub_manager"])
    hardware_platform.set_pub_que_dict(pub.get_pub_que_dict())

    sub.start()
    pub.start()
    try:
        hardware_platform.initialize()
        hardware_platform.start()
    finally:
        # Stop torque control even if something above raised.
        hardware_platform.shutdown()
        pub.stop()
        sub.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, parse_nested_args(parse_unknown_args(unknown)))
    print(OmegaConf.to_yaml(cfg))

    run_hardware(cfg)
