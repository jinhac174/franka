"""Hardware platform factory."""

from omegaconf import DictConfig

from robot_platform.hardware.BaseHardwarePlatform import BaseHardwarePlatform


def get_hardware_platform(cfg_platform: DictConfig) -> BaseHardwarePlatform:
    """Create a hardware platform from ``cfg_platform['name']``."""
    name = cfg_platform["name"]

    if name == "franka_fr3":
        from robot_platform.hardware.franka_fr3.HardwareFrankaFR3 import (
            HardwareFrankaFR3,
        )

        return HardwareFrankaFR3(cfg_platform)

    raise ValueError(f"Unknown hardware platform name: {name}")
