from omegaconf import DictConfig

from communication.lcm.publisher.BasePubManager import BasePubManager


def get_hardware_pub_manager(cfg_pub_manager: DictConfig) -> BasePubManager:
    """Get the hardware-side publisher manager (same messages as the sim env)."""
    name = cfg_pub_manager["name"]

    if name == "franka":
        from communication.lcm.publisher.pub_manager.env.franka_fr3.FrankaPubManager import (
            FrankaPubManager,
        )

        return FrankaPubManager(**cfg_pub_manager)

    raise ValueError(f"Publisher manager {name} not found.")
