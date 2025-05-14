from __future__ import annotations

import json
import os
import random
import time

import numpy as np

import tampura_environments.panda_utils.pb_utils as pbu
from tampura_environments.panda_utils.panda_env_utils import (
    CLIENT_MAP, EXCLUDE_CLASSES, World, all_ycb_names, create_ycb, get_grasp,
    get_ycb_obj_path, setup_robot_pybullet)
from tampura_environments.panda_utils.robot import DEFAULT_ARM_POS, PandaRobot


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == "__main__":
    robot_body, client = setup_robot_pybullet(gui=False)

    robot = PandaRobot(robot_body, client=client)

    client_id = len(CLIENT_MAP)
    CLIENT_MAP[client_id] = client

    arm_joint_names = ["panda_joint{}".format(i) for i in range(1, 8)]
    pbu.set_joint_positions(
        robot,
        pbu.joints_from_names(robot, arm_joint_names, client=client),
        DEFAULT_ARM_POS,
        client=client,
    )
    world = World(client_id=client_id, robot=robot, environment=[], floor=None)

    while True:
        names = all_ycb_names()
        names = [name for name in names if "banana" in name]
        names = [name for name in names if name not in EXCLUDE_CLASSES]
        print(names)
        random.shuffle(names)
        for cat in names:
            obj = create_ycb(cat, client=client)
            ycb_path = get_ycb_obj_path(cat)
            fp = os.path.join(os.path.dirname(ycb_path), "grasps")
            create_folder(fp)

            pbu.set_pose(
                obj,
                pbu.Pose(pbu.Point(0.5, 0.5, 0.5), pbu.Euler(pitch=np.pi / 2.0)),
                client=client,
            )
            grasp = get_grasp(world, obj, client=world.client)

            if grasp is not None:
                data = {"grasp": grasp.grasp}
                filename = "{}.json".format(str(time.time()))
                with open(os.path.join(fp, filename), "w") as f:
                    json.dump(data, f)

            pbu.remove_body(obj, client=client)
