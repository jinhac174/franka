"""Environment factory."""

from omegaconf import DictConfig

from env.BaseEnv import BaseEnv


def get_env(cfg_env: DictConfig) -> BaseEnv:
    """Create an environment from ``cfg_env['name']``."""
    name = cfg_env["name"]

    if name == "default_mujoco":
        from env.MujocoBaseEnv import MujocoBaseEnv

        return MujocoBaseEnv(cfg_env)

    if name == "mujoco_col":
        from env.MujocoColEnv import MujocoColEnv

        return MujocoColEnv(cfg_env)

    if name == "mujoco_cam":
        from env.MujocoCamEnv import MujocoCamEnv

        return MujocoCamEnv(cfg_env)

    if name == "mujoco_grasp_cam":
        from env.MujocoGraspEnv import MujocoGraspCamEnv

        return MujocoGraspCamEnv(cfg_env)

    if name == "mujoco_grasp":
        from env.MujocoGraspEnv import MujocoGraspEnv

        return MujocoGraspEnv(cfg_env)

    if name == "table_mesh_cam":
        from env.perception_tutorial.TableMeshCamEnv import TableMeshCamEnv

        return TableMeshCamEnv(cfg_env)

    raise ValueError(f"Unknown environment name: {name}.")
