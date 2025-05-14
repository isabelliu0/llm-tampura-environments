from __future__ import annotations

import json
import math
import os
import random
import sys
import time

from tampura_environments.panda_utils.robot import PandaRobot

ROOT_PATH = os.path.abspath(os.path.join(__file__, *[os.pardir] * 3))
SRL_PATH = os.path.join(ROOT_PATH, "models/srl")
PANDA_PATH = os.path.join(
    ROOT_PATH, "models/srl/franka_description/robots/panda_arm_hand.urdf"
)
import shutil

import numpy as np

import tampura_environments.panda_utils.pb_utils as pbu
from tampura_environments.panda_utils.panda_env_utils import (
    EXCLUDE_CLASSES, GRIPPER_GROUP, OPEN_GRIPPER_POS, all_ycb_names,
    create_default_env, create_ycb, setup_robot_pybullet)
from tampura_environments.panda_utils.primitives import GroupConf
from tampura_environments.panda_utils.robot import DEFAULT_ARM_POS


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
    dir_path = "./tampura_environments/find_dice/problems"
    os.makedirs(dir_path, exist_ok=True)

    MAX_ITER = 20
    MAX_ATTEMPTS = 20

    while True:
        robot_body, client = setup_robot_pybullet(gui=False)

        robot = PandaRobot(robot_body, client=client)

        floor, obstacles = create_default_env(client=client)

        arm_joint_names = ["panda_joint{}".format(i) for i in range(1, 8)]

        pbu.set_joint_positions(
            robot,
            pbu.joints_from_names(robot, arm_joint_names, client=client),
            DEFAULT_ARM_POS,
            client=client,
        )

        empty_camera_image = robot.get_image(client=client)

        open_gripper_conf = GroupConf(
            body=robot, group=GRIPPER_GROUP, positions=OPEN_GRIPPER_POS, client=client
        )
        open_gripper_conf.assign(client=client)

        link_pose = pbu.get_link_pose(
            robot,
            pbu.link_from_name(robot, "camera_frame", client=client),
            client=client,
        )
        link_point = list(link_pose[0])
        link_point[0] += 0.02

        strawberry_occluded = False

        faabb = pbu.get_aabb(floor, client=client)
        num_objects = random.randint(1, 5)

        INCLUDE_CLASSES = [
            "a_cups_ud",
            "b_cups_ud",
            "c_cups_ud",
            "d_cups_ud",
            "e_cups_ud",
        ]
        all_cats = INCLUDE_CLASSES

        # all_cats = list(set(all_ycb_names()) - {"dice"} - set(EXCLUDE_CLASSES))

        random.shuffle(all_cats)
        categories = ["dice"] + all_cats[:num_objects]
        random.shuffle(categories)

        objs = [create_ycb(cat, client=client, use_concave=True) for cat in categories]
        sample_objects_counter = 0
        while not strawberry_occluded:
            obj_map = {}
            all_objects = []
            all_poses = []

            success = True
            cup_pose = None
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

                    if "cups_ud" in cat:
                        cup_pose = pose

                    if "dice" in cat and cup_pose is not None:
                        pose = pbu.Pose(cup_pose[0])

                    pbu.set_pose(obj, pose, client=client)

                    width, height = pbu.dimensions_from_camera_matrix(
                        empty_camera_image.camera_matrix
                    )
                    p_in_cam = pbu.tform_point(
                        pbu.invert(empty_camera_image.camera_pose), point
                    )
                    pixel = pbu.pixel_from_point(
                        empty_camera_image.camera_matrix, p_in_cam
                    )

                    if (
                        any(
                            pbu.pairwise_collision(o, obj, client=client)
                            for o in all_objects
                        )
                        or pixel is None
                    ):
                        continue
                    else:
                        break
                else:
                    success = False
                    break

                all_objects.append(obj)
                all_poses.append(pose)

            if not success:
                sample_objects_counter += 1
                if sample_objects_counter > MAX_ATTEMPTS:
                    break
                continue
            else:
                camera_image = robot.get_image(client=client)
                id_image = camera_image.segmentationMaskBuffer[:, :, 0]
                unique_ids = np.unique(id_image)

                if obj_map["dice"] in unique_ids:
                    sample_objects_counter += 1
                    if sample_objects_counter > MAX_ATTEMPTS:
                        break
                else:
                    print("Didn't find it. Saving now...")
                    data_dict = {"categories": categories, "poses": all_poses}
                    # Saving the dictionary as a JSON file
                    fn = str(time.time()) + ".json"
                    with open(os.path.join(dir_path, fn), "w") as json_file:
                        json.dump(data_dict, json_file, cls=NumpyEncoder)

                    strawberry_occluded = True
        client.disconnect()
