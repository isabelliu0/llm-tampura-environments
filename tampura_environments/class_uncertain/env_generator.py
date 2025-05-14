from __future__ import annotations

import json
import math
import os
import random
import time

from tampura_environments.class_uncertain.env import (BALL_CLASSES, BOWL_CLASS,
                                                      FRUIT_CLASSES)

ROOT_PATH = os.path.abspath(os.path.join(__file__, *[os.pardir] * 3))
SRL_PATH = os.path.join(ROOT_PATH, "models/srl")
BOWL_SCALE = 1.5

import shutil

import numpy as np

import tampura_environments.panda_utils.pb_utils as pbu
from tampura_environments.panda_utils.panda_env_utils import (
    create_default_env, create_ycb, setup_robot_pybullet)


# Custom JSON encoder to handle NumPy ndarrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def create_empty_directory(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)

    os.makedirs(directory_path)


if __name__ == "__main__":
    dir_path = "./tampura_environments/class_uncertain/problems"

    MAX_ITER = 20
    NUM_PROBLEMS = 20
    for problem in range(NUM_PROBLEMS):
        robot_body, client = setup_robot_pybullet(gui=False)
        floor, obstacles = create_default_env(client=client)
        faabb = pbu.get_aabb(floor, client=client)

        num_fruit_objects = random.randint(1, 3)
        num_nonfruit_objects = 3 - num_fruit_objects

        fruit_cats = list(np.random.choice(FRUIT_CLASSES, size=num_fruit_objects))
        nonfruit_cats = list(np.random.choice(BALL_CLASSES, size=num_nonfruit_objects))
        fruits = [
            create_ycb(cat, client=client, use_concave=True) for cat in fruit_cats
        ]
        non_fruits = [
            create_ycb(cat, client=client, use_concave=True) for cat in nonfruit_cats
        ]
        bowl = create_ycb(BOWL_CLASS, client=client, use_concave=True, scale=BOWL_SCALE)
        sample_objects_counter = 0

        obj_map = {}
        all_objects = []
        all_poses = []
        all_scales = []

        success = True
        categories = fruit_cats + nonfruit_cats + [BOWL_CLASS]
        objs = fruits + non_fruits + [bowl]
        all_scales = [1.0] * len(fruits + non_fruits) + [BOWL_SCALE]

        for cat, obj in zip(categories, objs):
            obj_map[cat] = obj
            for _ in range(MAX_ITER):
                point = pbu.Point(
                    x=random.uniform(0.3, 0.7),
                    y=random.uniform(-0.4, 0.4),
                    z=pbu.stable_z_on_aabb(obj, faabb, client=client) + 0.01,
                )
                euler = pbu.Euler(yaw=random.uniform(-math.pi, math.pi))
                pose = pbu.Pose(point=point, euler=euler)
                pbu.set_pose(obj, pose, client=client)

                if any(
                    pbu.pairwise_collision(o, obj, client=client) for o in all_objects
                ):
                    continue
                else:
                    break
            else:
                success = False
                break
            all_objects.append(obj)
            all_poses.append(pose)

        client.disconnect()
        if success:
            data_dict = {
                "categories": categories,
                "poses": all_poses,
                "scales": all_scales,
            }
            # Saving the dictionary as a JSON file
            fn = str(time.time()) + ".json"
            with open(os.path.join(dir_path, fn), "w") as json_file:
                json.dump(data_dict, json_file, cls=NumpyEncoder)
