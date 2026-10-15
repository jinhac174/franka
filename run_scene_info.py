"""Publish collision geometry for the real robot (no simulator running).

In simulation, ``MujocoColEnv`` publishes the robot's planning primitives and
the cell's static obstacles (floor / walls / table / obstacles). On the real
FR3, run_hardware.py does it (``scene_info:`` in its config); this process is
the stand-alone version (e.g. to try a scene with the fake robot): it reads the same kind of
scene XML (describe YOUR cell in it: ``floor``, ``wall_*``, ``table*``,
``obs_*`` geoms) and republishes the geometry periodically.

    python run_scene_info.py --config configs/franka_fr3/hardware/scene_info.yaml
"""

import argparse

from omegaconf import OmegaConf

from robot_platform.hardware.franka_fr3.scene_info import SceneInfoPublisher
from utils.terminal.argparse_utils import parse_nested_args, parse_unknown_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), parse_nested_args(parse_unknown_args(unknown)))
    print(OmegaConf.to_yaml(cfg))

    pub = SceneInfoPublisher(cfg["scene_xml_path"], cfg["static_col_info_channel"],
                             cfg["robot_col_info_channel"], cfg.get("rate", 10.0))
    print(f"[SceneInfo] {pub.describe()}")
    try:
        pub.run()
    except KeyboardInterrupt:
        print("[SceneInfo] stopped.")


if __name__ == "__main__":
    main()
