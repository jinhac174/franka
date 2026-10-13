"""Publish collision geometry for the real robot (no simulator running).

In simulation, ``MujocoColEnv`` publishes the robot's planning primitives and
the cell's static obstacles (floor / walls / table / obstacles). On the real
FR3 there is no env, so this process does that job: it reads the same kind of
scene XML (describe YOUR cell in it: ``floor``, ``wall_*``, ``table*``,
``obs_*`` geoms) and republishes the geometry periodically.

    python run_scene_info.py --config configs/franka_fr3/hardware/scene_info.yaml
"""

import argparse
import time
from queue import Queue

import lcm
import mujoco as mj
from omegaconf import OmegaConf

from communication.lcm.publisher.data_publisher.ColInfoPublisher import ColInfoPublisher
from data_type.basic_types.ColInfoData import ColInfoData
from robot_platform.sim.franka_fr3.SimFrankaFR3 import fr3_col_primitives
from utils.collision.wall_geoms import extract_static_col_geoms
from utils.terminal.argparse_utils import parse_nested_args, parse_unknown_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), parse_nested_args(parse_unknown_args(unknown)))
    print(OmegaConf.to_yaml(cfg))

    model = mj.MjModel.from_xml_path(cfg["scene_xml_path"])
    static = ColInfoData(name=cfg["static_col_info_channel"])
    for g in extract_static_col_geoms(model):
        static.add_col_geom(**g)
    robot = ColInfoData(name=cfg["robot_col_info_channel"])
    for g in fr3_col_primitives():
        robot.add_col_geom(**g)
    print(f"[SceneInfo] static: {sorted(static.dict_col_geoms)}; robot: {len(robot.dict_col_geoms)} primitives")

    lc = lcm.LCM()
    pubs = []
    for data in (static, robot):
        que = Queue()
        pubs.append((ColInfoPublisher(lc, data.name, que), que, data))
    period = 1.0 / float(cfg.get("rate", 10.0))
    try:
        while True:
            t = time.time()
            for pub, que, data in pubs:
                data.set_time(t)
                que.put(data)
                pub.check_que_and_publish()
            time.sleep(period)
    except KeyboardInterrupt:
        print("[SceneInfo] stopped.")


if __name__ == "__main__":
    main()
