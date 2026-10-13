from omegaconf import DictConfig

from communication.lcm.subscriber.BaseSubManager import (
    BaseEnvSubManeager,
    BaseSubManager,
)


def get_env_sub_manager(cfg_sub_manager: DictConfig) -> BaseSubManager:
    """Get the subscriber manager based on the configuration.
    @param[in] cfg_sub_manager: Configuration for the subscriber manager.
    @return: Subscriber manager object.
    """
    name = cfg_sub_manager["name"]

    if name == "base":
        return BaseEnvSubManeager(**cfg_sub_manager)
    elif name == "franka":
        from communication.lcm.subscriber.sub_manager.env.FrankaSubManager import (
            FrankaSubManager,
        )

        return FrankaSubManager(**cfg_sub_manager)

    raise ValueError(f"Unknown subscriber manager: {name}")
