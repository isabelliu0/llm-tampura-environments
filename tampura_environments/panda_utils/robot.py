import os
import time
from collections import OrderedDict

import numpy as np

import tampura_environments.panda_utils.pb_utils as pbu

CAMERA_FRAME = "camera_frame"
CAMERA_OPTICAL_FRAME = "camera_frame"

WIDTH, HEIGHT = 640, 480
FX, FY = 525.0, 525.0

# Lower resolution camera params
# WIDTH, HEIGHT = 640 // 2, 480 // 2
# FX, FY = 525.0 / 4, 525.0 / 4

CAMERA_MATRIX = pbu.get_camera_matrix(WIDTH, HEIGHT, FX, FY)

PANDA_TOOL_TIP = "panda_tool_tip"
ARM_GROUP = "main_arm"
GRIPPER_GROUP = "main_gripper"
PANDA_GROUPS = {
    "base": [],
    "main_arm": ["panda_joint{}".format(i) for i in range(1, 8)],
    "main_gripper": ["panda_finger_joint1", "panda_finger_joint2"],
}

ROOT_PATH = os.path.abspath(os.path.join(__file__, *[os.pardir] * 3))
SRL_PATH = os.path.join(ROOT_PATH, "models/srl")
PANDA_PATH = os.path.join(ROOT_PATH, "models/srl/franka_panda/panda.urdf")
MISC_PATH = os.path.join(ROOT_PATH, "models/misc")
YCB_PATH = os.path.join(SRL_PATH, "ycb")

DEFAULT_ARM_POS = [
    -0.0806406098426434,
    -1.6722951504174777,
    0.07069076842695393,
    -2.7449419709102822,
    0.08184716251979611,
    1.7516337599063168,
    0.7849295270972781,
]


OPEN_GRIPPER_POS = [0.045, 0.045]
CLOSE_GRIPPER_POS = [0, 0]

# Fixed hand to depth transform for the real panda robot
HAND_TO_DEPTH = (
    (0.03649966282414946, -0.034889795700641386, 0.0574),
    (0.00252743, 0.0065769, 0.70345566, 0.71070423),
)


def save_camera_images(
    camera_image, directory="", prefix="", predicted=True, client=None, **kwargs
):
    pbu.ensure_dir(directory)
    rgb_image, depth_image, seg_image = camera_image[:3]
    pbu.save_image(
        os.path.join(directory, "{}rgb.png".format(prefix)), rgb_image
    )  # [0, 255]
    depth_image = (
        (depth_image - np.min(depth_image))
        / (np.max(depth_image) - np.min(depth_image))
        * 255
    ).astype(np.uint8)
    pbu.save_image(
        os.path.join(directory, "{}depth.png".format(prefix)), depth_image
    )  # [0, 1]
    if seg_image is None:
        return None

    segmented_image = pbu.image_from_segmented(seg_image, client=client)
    pbu.save_image(
        os.path.join(directory, "{}segmented.png".format(prefix)), segmented_image
    )
    return segmented_image


class Camera(object):  # TODO: extend Object?
    def __init__(
        self, robot, link, optical_frame, camera_matrix, max_depth=2.5, **kwargs
    ):
        self.robot = robot
        self.optical_frame = optical_frame
        self.camera_matrix = camera_matrix
        self.max_depth = max_depth
        self.kwargs = dict(kwargs)

    def get_pose(self, **kwargs):
        return pbu.get_link_pose(self.robot, self.optical_frame, **kwargs)

    def get_image(self, segment=True, segment_links=False, **kwargs):
        return pbu.get_image_at_pose(
            self.get_pose(**kwargs),
            self.camera_matrix,
            tiny=False,
            segment=segment,
            segment_links=segment_links,
            **kwargs,
        )


class PandaRobot:
    def __init__(
        self,
        robot_body,
        link_names={},
        camera_matrix=CAMERA_MATRIX,
        **kwargs,
    ):
        self.link_names = link_names
        self.body = robot_body
        self.joint_groups = PANDA_GROUPS
        self.components = {}

        self.camera = Camera(
            self,
            link=pbu.link_from_name(self.body, CAMERA_FRAME, **kwargs),
            optical_frame=pbu.link_from_name(self.body, CAMERA_OPTICAL_FRAME, **kwargs),
            camera_matrix=camera_matrix,
        )

        self.max_depth = 3.0
        self.min_z = 0.0
        self.BASE_LINK = "panda_link0"
        self.MAX_PANDA_FINGER = 0.045

        self.reset(**kwargs)

    def __int__(self):
        return self.body

    def get_default_conf(self):
        conf = {
            "main_arm": DEFAULT_ARM_POS,
            "main_gripper": [self.MAX_PANDA_FINGER, self.MAX_PANDA_FINGER],
        }
        return conf

    def arm_conf(self, arm, config):
        return config

    def get_closed_positions(self):
        return {"panda_finger_joint1": 0, "panda_finger_joint2": 0}

    def get_open_positions(self):
        return {
            "panda_finger_joint1": self.MAX_PANDA_FINGER,
            "panda_finger_joint2": self.MAX_PANDA_FINGER,
        }

    def get_group_joints(self, group, **kwargs):
        return pbu.joints_from_names(self.body, PANDA_GROUPS[group], **kwargs)

    def reset(self, **kwargs):
        conf = self.get_default_conf()
        for group, positions in conf.items():
            self.set_group_positions(group, positions, **kwargs)

    def get_group_limits(self, group, **kwargs):
        return pbu.get_custom_limits(
            self.body, self.get_group_joints(group, **kwargs), **kwargs
        )

    def get_gripper_width(self, gripper_joints, **kwargs):
        [link1, link2] = self.get_finger_links(gripper_joints, **kwargs)
        [collision_info] = pbu.get_closest_points(
            self.body, self.body, link1, link2, max_distance=np.inf, **kwargs
        )
        point1 = collision_info.positionOnA
        point2 = collision_info.positionOnB
        max_width = pbu.get_distance(point1, point2)
        return max_width

    def get_max_gripper_width(self, gripper_joints, **kwargs):
        with pbu.ConfSaver(self, **kwargs):
            pbu.set_joint_positions(
                self.body,
                gripper_joints,
                pbu.get_max_limits(self.body, gripper_joints, **kwargs),
                **kwargs,
            )
            return self.get_gripper_width(gripper_joints, **kwargs)

    def get_finger_links(self, gripper_joints, **kwargs):
        moving_links = pbu.get_moving_links(self.body, gripper_joints, **kwargs)
        shape_links = [
            link
            for link in moving_links
            if pbu.get_collision_data(self.body, link, **kwargs)
        ]
        finger_links = [
            link
            for link in shape_links
            if not any(
                pbu.get_collision_data(self.body, child, **kwargs)
                for child in pbu.get_link_children(self.body, link, **kwargs)
            )
        ]
        if len(finger_links) != 2:
            raise RuntimeError(finger_links)
        return finger_links

    def get_group_parent(self, group, **kwargs):
        return pbu.get_link_parent(
            self.body, self.get_group_joints(group, **kwargs)[0], **kwargs
        )

    def get_tool_link_pose(self, **kwargs):
        tool_link = pbu.link_from_name(self.body, PANDA_TOOL_TIP, **kwargs)
        return pbu.get_link_pose(self.body, tool_link, **kwargs)

    def get_parent_from_tool(self, **kwargs):
        tool_tip_link = pbu.link_from_name(self.body, PANDA_TOOL_TIP, **kwargs)
        parent_link = self.get_group_parent(GRIPPER_GROUP, **kwargs)
        return pbu.get_relative_pose(self.body, tool_tip_link, parent_link, **kwargs)

    def get_component_mapping(self, group, **kwargs):
        assert group in self.components
        component_joints = pbu.get_movable_joints(
            self.components[group], draw=False, **kwargs
        )
        body_joints = pbu.get_movable_joint_descendants(
            self.body, self.get_group_parent(group, **kwargs), **kwargs
        )
        return OrderedDict(pbu.safe_zip(body_joints, component_joints))

    def get_component_joints(self, group, **kwargs):
        mapping = self.get_component_mapping(group, **kwargs)
        return list(map(mapping.get, self.get_group_joints(group, **kwargs)))

    def get_component_info(self, fn, group, **kwargs):
        return fn(self.body, self.get_group_joints(group, **kwargs))

    def get_group_subtree(self, group, **kwargs):
        return pbu.get_link_subtree(
            self.body, self.get_group_parent(group, **kwargs), **kwargs
        )

    def get_component(self, group, visual=True, **kwargs):
        if group not in self.components:
            component = pbu.clone_body(
                self.body,
                links=self.get_group_subtree(group, **kwargs),
                visual=False,
                collision=True,
                **kwargs,
            )
            if not visual:
                pbu.set_all_color(component, pbu.TRANSPARENT)
            self.components[group] = component
        return self.components[group]

    def remove_components(self, **kwargs):
        for component in self.components.values():
            pbu.remove_body(component, **kwargs)
        self.components = {}

    def get_image(self, **kwargs):
        return self.camera.get_image(**kwargs)

    def get_group_joints(self, group, **kwargs):
        return pbu.joints_from_names(self.body, self.joint_groups[group], **kwargs)

    def set_group_conf(self, group, positions, **kwargs):
        pbu.set_joint_positions(
            self.body, self.get_group_joints(group, **kwargs), positions, **kwargs
        )

    def set_group_positions(self, group, positions, **kwargs):
        self.set_group_conf(group, positions, **kwargs)

    def get_joint_positions(self, **kwargs):
        joints = pbu.get_joints(self.body, **kwargs)
        joint_positions = pbu.get_joint_positions(self.body, joints, **kwargs)
        joint_names = pbu.get_joint_names(self.body, joints, **kwargs)
        return {k: v for k, v in zip(joint_names, joint_positions)}

    def command_group(self, group, positions, **kwargs):  # TODO: default timeout
        self.set_group_positions(group, positions, **kwargs)

    def command_group_dict(
        self, group, positions_dict, **kwargs
    ):  # TODO: default timeout
        positions = [positions_dict[nm] for nm in self.joint_groups[group]]
        self.command_group(group, positions, **kwargs)

    def command_group_trajectory(self, group, positions, dt=0.01, **kwargs):
        for position in positions:
            self.command_group(group, position, **kwargs)
            time.sleep(dt)

    def wait(self, duration):
        time.sleep(duration)

    def any_arm_fully_closed(self):
        return False
