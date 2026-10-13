"""Factory for MuJoCo simulation platforms."""

from omegaconf import DictConfig, OmegaConf

from robot_platform.sim.BaseSimPlatform import BaseSimPlatform


def get_sim_platform(
    cfg_platform: DictConfig,
    cfg_sub_manager: DictConfig = OmegaConf.create(),
    cfg_pub_manager: DictConfig = OmegaConf.create(),
) -> BaseSimPlatform:
    """Build a sim platform from config.

    Channel names from sub/pub manager configs are merged into the platform
    config so constructors can pick them up as kwargs.
    """
    name = cfg_platform["name"]
    cfg = OmegaConf.merge(cfg_platform, cfg_sub_manager, cfg_pub_manager)

    if name == "empty":
        from robot_platform.sim.EmptySimPlatform import EmptySimPlatform

        return EmptySimPlatform(**cfg)

    if name == "franka_fr3":
        from robot_platform.sim.franka_fr3.SimFrankaFR3 import SimFrankaFR3

        return SimFrankaFR3(**cfg)

    raise ValueError(f"Unknown platform name: {name}")
