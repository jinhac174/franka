from omegaconf import DictConfig

from communication.lcm.subscriber.BaseSubManager import BaseSubManager


def get_hardware_sub_manager(cfg_sub_manager: DictConfig) -> BaseSubManager:
    """Get the hardware-side subscriber manager (same messages as the sim env)."""
    name = cfg_sub_manager["name"]

    if name == "franka":
        from communication.lcm.subscriber.sub_manager.env.FrankaSubManager import (
            FrankaSubManager,
        )

        return FrankaSubManager(**cfg_sub_manager)

    raise ValueError(f"Unknown subscriber manager: {name}")
