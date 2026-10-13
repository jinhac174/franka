"""
This module provides a function to create a visualizer manager based on the configuration provided.
"""

from omegaconf import DictConfig
from visualizer.BaseVisManager import BaseVisManager


def get_vis_manager(cfg_vis_manager: DictConfig) -> BaseVisManager:
    """
    Get the visualizer manager based on the configuration provided.
    @param[in] cfg_vis_manager: Configuration for the visualizer manager.
    """
    if cfg_vis_manager["name"] == "base":
        return BaseVisManager(cfg_vis_manager)
    elif cfg_vis_manager["name"].startswith("franka_fr3"):
        from visualizer.vis_manager.franka_fr3.FrankaVisManager import (
            FrankaVisManager,
        )

        return FrankaVisManager(cfg_vis_manager)
    elif cfg_vis_manager["name"] == "obj_pose":
        from visualizer.vis_manager.perception_tutorial.ObjPoseVisManager import (
            ObjPoseVisManager,
        )

        return ObjPoseVisManager(cfg_vis_manager)
    else:
        raise ValueError(f"Visualizer manager {cfg_vis_manager['name']} not found.")
