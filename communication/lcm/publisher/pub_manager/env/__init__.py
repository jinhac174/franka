from omegaconf import DictConfig

from communication.lcm.publisher.BasePubManager import BasePubManager


def get_env_pub_manager(cfg_pub_manager: DictConfig) -> BasePubManager:
    """Get the publisher manager based on the configuration.
    @param[in] cfg_pub_manager: Configuration for the publisher manager.
    @return: An instance of the publisher manager.
    """
    name = cfg_pub_manager["name"]

    if name == "base":
        return BasePubManager(**cfg_pub_manager)

    elif name == "franka":
        from communication.lcm.publisher.pub_manager.env.franka_fr3.FrankaPubManager import (
            FrankaPubManager,
        )

        return FrankaPubManager(**cfg_pub_manager)

    elif name == "cam_only":
        from communication.lcm.publisher.pub_manager.env.CamOnlyPubManager import (
            CamOnlyPubManager,
        )

        return CamOnlyPubManager(**cfg_pub_manager)

    raise ValueError(f"Publisher manager {name} not found.")
