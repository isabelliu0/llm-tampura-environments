from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p  # type:ignore
import pybullet_utils.bullet_client as bc  # type:ignore
import trimesh  # type:ignore
from scipy.spatial.transform import Rotation as R  # type:ignore
from tampura.structs import State  # type:ignore

import tampura_environments.panda_utils.pb_utils as pbu
from tampura_environments.panda_utils.grasping import (Z_AXIS, Plane,
                                                       generate_mesh_grasps,
                                                       get_postgrasp,
                                                       sorted_grasps)
from tampura_environments.panda_utils.motion_planning.motion_planners.meta import \
    birrt
from tampura_environments.panda_utils.primitives import (Command, Grasp,
                                                         GroupConf,
                                                         GroupTrajectory)
from tampura_environments.panda_utils.robot import (ARM_GROUP, CAMERA_MATRIX,
                                                    GRIPPER_GROUP,
                                                    OPEN_GRIPPER_POS,
                                                    PANDA_PATH, PANDA_TOOL_TIP,
                                                    YCB_PATH, PandaRobot)

MAX_IK_TIME = 0.01
MAX_IK_DISTANCE = np.inf
MAX_TOOL_DISTANCE = np.inf
DISABLE_ALL_COLLISIONS = True
COLLISION_EPSILON = 1e-3
COLLISION_DISTANCE = 5e-3  # Distance from fixed obstacles
SELF_COLLISIONS = True
EPSILON = 1e-3

TOOL_POSE = pbu.Pose(
    point=pbu.Point(x=0.00), euler=pbu.Euler(pitch=np.pi / 2)
)  # +x out of gripper arm

SWITCH_BEFORE = "grasp"  # contact | grasp | pregrasp | arm | none
BASE_COST = 1
PROXIMITY_COST_TERM = False
RELAX_GRASP_COLLISIONS = False
GRASP_EXPERIMENT = False

GEOMETRIC_MODES = ["top", "side", "mesh"]
LEARNED_MODES = ["gpd", "graspnet"]
MODE_ORDERS = ["", "_random", "_best"]


PYBULLET_YCB_DIR = "pybullet-object-models/pybullet_object_models/ycb_objects"

EXCLUDE_CLASSES = [
    "fork",
    "master_chef_can",
    "skilled_lid",
    "chain",
    "wood_block",
    "mini_soccer_ball",
    "a_colored_wood_blocks",
    "softball",
    "i_cups",
    "j_cups",
    "i_cups_ud",
    "j_cups_ud",
    "pudding_box",
]
GRIPPER_RES = 0.001
GRID_RESOLUTION = 0.02

CLIENT_MAP: Dict[int, Any] = {}


def pose_to_vec(pose: pbu.PoseType) -> np.ndarray:
    return np.array(list(pose[0]) + list(pose[1]))


def transformation_to_pose(trans):
    matrix = np.array(trans)

    # Extract the rotation matrix (top-left 3x3)
    rotation_matrix = matrix[:3, :3]

    # Extract the translation vector (first three elements of the fourth column)
    translation_vector = matrix[:3, 3]

    # Convert the rotation matrix to a quaternion
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    return (translation_vector, quaternion)


def pose_to_transformation(pose):
    # Convert the quaternion to a rotation matrix
    rotation_matrix = R.from_quat(pose[1]).as_matrix()

    # Create the transformation matrix
    transformation_matrix = np.eye(4)  # Initialize a 4x4 identity matrix
    transformation_matrix[:3, :3] = (
        rotation_matrix  # Set the top-left 3x3 to the rotation matrix
    )
    transformation_matrix[:3, 3] = pose[
        0
    ]  # Set the top three elements of the fourth column to the translation vector

    return transformation_matrix


def transform_points(matrix, points):
    # Convert points to homogeneous coordinates (add a column of ones)
    num_points = points.shape[0]
    points_homogeneous = np.hstack((points, np.ones((num_points, 1))))

    # Apply the transformation matrix
    transformed_points_homogeneous = np.dot(points_homogeneous, matrix.T)

    # Convert back to Cartesian coordinates
    transformed_points = (
        transformed_points_homogeneous[:, :3] / transformed_points_homogeneous[:, [3]]
    )

    return transformed_points


def plan_workspace_motion(
    robot: PandaRobot,
    tool_waypoints,
    attachment=None,
    obstacles=[],
    max_attempts=20,
    **kwargs,
):
    assert tool_waypoints

    tool_link = pbu.link_from_name(robot, PANDA_TOOL_TIP, **kwargs)
    arm_joints = pbu.joints_from_names(robot, robot.joint_groups[ARM_GROUP], **kwargs)
    parts = [robot] + ([] if attachment is None else [attachment.child])

    collision_fn = pbu.get_collision_fn(
        robot,
        arm_joints,
        obstacles=obstacles,
        attachments=[],
        self_collisions=SELF_COLLISIONS,
        disable_collisions=True,
        **kwargs,
    )

    for attempts in range(max_attempts):
        if attempts > 0:
            shrink = 0.2
            ranges = [
                pbu.get_joint_limits(robot, joint, **kwargs) for joint in arm_joints
            ]
            initialization_sample = []
            for r in ranges:
                mid = (r.lower + r.upper) / 2.0
                shrink_lower = mid - (shrink * (r.upper - r.lower) / 2)
                shrink_upper = mid + (shrink * (r.upper - r.lower) / 2)
                initialization_sample.append(random.uniform(shrink_lower, shrink_upper))
            pbu.set_joint_positions(robot, arm_joints, initialization_sample, **kwargs)

        arm_conf = pbu.inverse_kinematics(
            robot, tool_link, tool_waypoints[0], arm_joints, max_iterations=1, **kwargs
        )

        if arm_conf is None:
            continue

        if collision_fn(arm_conf):
            continue

        arm_waypoints = [arm_conf]

        for tool_pose in tool_waypoints[1:]:
            arm_conf = pbu.inverse_kinematics(
                robot, tool_link, tool_pose, arm_joints, max_iterations=1, **kwargs
            )

            if arm_conf is None:
                break

            if collision_fn(arm_conf):
                break

            arm_waypoints.append(arm_conf)

        else:
            pbu.set_joint_positions(robot, arm_joints, arm_waypoints[-1], **kwargs)
            if attachment is not None:
                attachment.assign()
            if (
                any(
                    pbu.pairwise_collisions(
                        part,
                        obstacles,
                        max_distance=(COLLISION_DISTANCE + EPSILON),
                        **kwargs,
                    )
                    for part in parts
                )
                and not DISABLE_ALL_COLLISIONS
            ):
                continue
            arm_path = pbu.interpolate_joint_waypoints(
                robot, arm_joints, arm_waypoints, **kwargs
            )

            if any(collision_fn(q) for q in arm_path):
                continue

            return arm_path
    return None


def z_plane(z=0.0):
    normal = Z_AXIS
    origin = z * normal
    return Plane(normal, origin)


def close_until_collision(
    robot,
    gripper_joints,
    gripper_group,
    bodies=[],
    open_conf=None,
    closed_conf=None,
    num_steps=25,
    **kwargs,
):
    if not gripper_joints:
        return None

    closed_conf, open_conf = robot.get_group_limits(gripper_group, **kwargs)
    resolutions = np.abs(np.array(open_conf) - np.array(closed_conf)) / num_steps
    extend_fn = pbu.get_extend_fn(
        robot, gripper_joints, resolutions=resolutions, **kwargs
    )
    close_path = [open_conf] + list(extend_fn(open_conf, closed_conf))
    collision_links = frozenset(pbu.get_moving_links(robot, gripper_joints, **kwargs))
    for i, conf in enumerate(close_path):
        pbu.set_joint_positions(robot, gripper_joints, conf, **kwargs)
        if any(
            pbu.pairwise_collision(
                pbu.CollisionPair(robot, collision_links), body, **kwargs
            )
            for body in bodies
        ):
            if i == 0:
                return None
            return close_path[i - 1][0]
    return close_path[-1][0]


def get_grasp_candidates(obj, gripper_width=np.inf, grasp_mode="mesh", **kwargs):
    if grasp_mode == "saved":
        [data] = pbu.get_visual_data(obj, -1, **kwargs)
        filename = pbu.get_data_filename(data)

        # get directory of filename
        grasp_dir = os.path.join(os.path.dirname(filename), "grasps")

        # Get list of files in grasp_dir that end with json
        if os.path.isdir(grasp_dir):
            grasp_files = [f for f in os.listdir(grasp_dir) if f.endswith(".json")]
        else:
            grasp_files = []

        # Load in all of the json files into one list
        grasp_list = []
        for grasp_file in grasp_files:
            with open(os.path.join(grasp_dir, grasp_file), "r") as f:
                grasp_list.append(json.load(f)["grasp"])
        random.shuffle(grasp_list)
        return grasp_list
    elif grasp_mode == "mesh":
        pitches = [-np.pi, np.pi]
        target_tolerance = np.pi / 4
        z_threshold = 1e-2
        antipodal_tolerance = np.pi / 16

        generated_grasps = generate_mesh_grasps(
            obj,
            pitches=pitches,
            discrete_pitch=False,
            max_width=gripper_width,
            max_time=2,
            target_tolerance=target_tolerance,
            antipodal_tolerance=antipodal_tolerance,
            z_threshold=z_threshold,
            **kwargs,
        )

        if generated_grasps is not None:
            return (
                grasp
                for grasp, contact1, contact2, score in sorted_grasps(
                    generated_grasps, max_candidates=10, p_random=0.0, **kwargs
                )
            )
        else:
            return tuple([])
    elif grasp_mode == "top":
        aabb, pose = pbu.get_oobb(obj, **kwargs)
        cand = pbu.get_top_and_bottom_grasps(
            obj, aabb, pose, tool_pose=TOOL_POSE, grasp_length=0.01, **kwargs
        )
        return random.sample(cand, len(cand))
        # return [multiply(Pose(euler=Euler(pitch=-np.pi / 2.0)), Pose(Point(z=-0.01)))]


def setup_robot(vis=False, camera_matrix=CAMERA_MATRIX) -> Tuple[PandaRobot, int]:
    robot_body, client = setup_robot_pybullet(gui=vis)

    robot = PandaRobot(
        robot_body,
        camera_matrix=camera_matrix,
        client=client,
    )

    return robot, client


#######################################################
def get_grasp(
    world,
    obj,
    obstacles=[],
    grasp_mode="mesh",
    gripper_collisions=True,
    closed_fraction=5e-2,
    max_attempts=float("inf"),
    **kwargs,
):
    closed_conf, open_conf = world.robot.get_group_limits(GRIPPER_GROUP, **kwargs)

    max_width = world.robot.get_max_gripper_width(
        world.robot.get_group_joints(GRIPPER_GROUP, **kwargs), **kwargs
    )
    gripper_width = max_width - 1e-2
    generator = iter(
        get_grasp_candidates(
            obj, grasp_mode=grasp_mode, gripper_width=gripper_width, **kwargs
        )
    )
    # Remember the starting finger positions to we can restore them at the end
    original_finger_positions = pbu.get_joint_positions(
        world.robot, world.robot.get_group_joints(GRIPPER_GROUP, **kwargs), **kwargs
    )

    # Remember the original object pose so we can reset it
    original_pose = pbu.get_pose(obj, **kwargs)

    gripper = world.robot.get_component(GRIPPER_GROUP, **kwargs)
    parent_from_tool = world.robot.get_parent_from_tool(**kwargs)

    # Time tracking.
    attempts = 0
    while True:
        grasp_pose = next(generator, None)
        # Handle termination conditions.
        if not grasp_pose or attempts >= max_attempts:
            logging.debug(f"Grasps for {obj} timed out after {attempts} attempts")
            world.robot.remove_components(**kwargs)
            return None

        attempts += 1

        # Set gripper and object poses.
        new_pose = pbu.multiply(
            pbu.get_pose(obj, **kwargs),
            pbu.invert(pbu.multiply(parent_from_tool, grasp_pose)),
        )
        pbu.set_pose(gripper, new_pose, **kwargs)

        pbu.set_joint_positions(
            gripper,
            world.robot.get_component_joints(GRIPPER_GROUP, **kwargs),
            open_conf,
            **kwargs,
        )

        obstacles = [obj] if gripper_collisions else []
        obstacles.extend(obstacles)

        # Check for collisions.
        if pbu.pairwise_collisions(gripper, obstacles, **kwargs):
            continue

        pbu.set_pose(
            obj,
            pbu.multiply(world.robot.get_tool_link_pose(**kwargs), grasp_pose),
            **kwargs,
        )

        # Check for pairwise collisions with the object.
        if pbu.pairwise_collision(gripper, obj, **kwargs):
            continue

        gripper_joints = world.robot.get_group_joints(GRIPPER_GROUP, **kwargs)

        closed_position = closed_conf[0]
        if gripper_collisions:
            closed_position = close_until_collision(
                world.robot,
                gripper_joints,
                GRIPPER_GROUP,
                bodies=[obj],
                max_distance=0.0,
                **kwargs,
            )
            if closed_position is None:
                continue

        closed_position = (1 + closed_fraction) * closed_position
        grasp = Grasp(obj, grasp_pose, closed_position=closed_position, **kwargs)

        # Restore original finger positions
        pbu.set_joint_positions(
            world.robot,
            world.robot.get_group_joints(GRIPPER_GROUP, **kwargs),
            original_finger_positions,
            **kwargs,
        )

        # Restore original object pose
        pbu.set_pose(obj, original_pose, **kwargs)
        world.robot.remove_components(**kwargs)
        return grasp


def plan_joint_motion(
    body,
    joints,
    end_conf,
    obstacles=[],
    attachments=[],
    self_collisions=True,
    disabled_collisions=set(),
    weights=None,
    resolutions=None,
    max_distance=pbu.MAX_DISTANCE,
    use_aabb=False,
    cache=True,
    custom_limits={},
    disable_collisions=False,
    extra_collisions=None,
    **kwargs,
):
    assert len(joints) == len(end_conf)
    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)
    sample_fn = pbu.get_sample_fn(body, joints, custom_limits=custom_limits, **kwargs)
    distance_fn = pbu.get_distance_fn(body, joints, weights=weights, **kwargs)
    extend_fn = pbu.get_extend_fn(body, joints, resolutions=resolutions, **kwargs)
    collision_fn = pbu.get_collision_fn(
        body,
        joints,
        obstacles,
        attachments,
        self_collisions,
        disabled_collisions,
        custom_limits=custom_limits,
        max_distance=max_distance,
        use_aabb=use_aabb,
        cache=cache,
        disable_collisions=disable_collisions,
        extra_collisions=extra_collisions,
        **kwargs,
    )

    start_conf = pbu.get_joint_positions(body, joints, **kwargs)
    if not pbu.check_initial_end(
        body, joints, start_conf, end_conf, collision_fn, **kwargs
    ):
        return None

    return birrt(
        start_conf,
        end_conf,
        distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        **kwargs,
    )


def plan_motion_fn(
    robot: PandaRobot,
    group: str,
    q1: GroupConf,
    q2: GroupConf,
    attachments: List[pbu.Attachment] = [],
    environment: List[int] = [],
    **kwargs,
) -> Optional[GroupTrajectory]:

    obstacles = list(environment)
    attached = {attachment.child for attachment in attachments}
    obstacles = list(set(obstacles) - set(attached))
    q1.assign(**kwargs)

    resolutions = math.radians(10) * np.ones(len(q2.joints))

    path = plan_joint_motion(
        robot,
        q2.joints,
        q2.positions,
        resolutions=resolutions,
        obstacles=obstacles,
        attachments=attachments,
        self_collisions=SELF_COLLISIONS,
        max_distance=COLLISION_DISTANCE,
        restarts=1,
        iterations=20,
        smooth=100,
        max_iterations=1000,
        disable_collisions=DISABLE_ALL_COLLISIONS,
        extra_collisions=None,
        **kwargs,
    )

    if path is None:
        for conf in [q1, q2]:
            conf.assign(**kwargs)
            for attachment in attachments:
                attachment.assign(**kwargs)
        return None

    q1.assign(**kwargs)
    for attachment in attachments:
        attachment.assign(**kwargs)

    return GroupTrajectory(robot, group, path, **kwargs)


def create_block(
    color: pbu.RGBA,
    position: pbu.PointType = (0, 0, 0.5),
    halfExtents: Tuple[float, float, float] = (0.02, 0.02, 0.02),
    client: Any = None,
) -> int:
    # Creating the visual shape with the specified color
    visualShapeId = client.createVisualShape(
        shapeType=client.GEOM_BOX, rgbaColor=color, halfExtents=halfExtents
    )

    # Creating the collision shape with the same parameters
    collisionShapeId = client.createCollisionShape(
        shapeType=client.GEOM_BOX, halfExtents=halfExtents
    )

    # Creating the cube (multi-body with both collision and visual shapes)
    bodyId = client.createMultiBody(
        baseMass=0.1,
        baseVisualShapeIndex=visualShapeId,
        baseCollisionShapeIndex=collisionShapeId,
        basePosition=position,  # static block
    )

    return bodyId


@dataclass
class World:
    client_id: int
    robot: PandaRobot
    environment: Any
    floor: Any
    objects: List[Any] = field(default_factory=lambda: [])
    categories: List[str] = field(default_factory=lambda: [])
    regions: List[str] = field(default_factory=lambda: [])

    @property
    def poses(self):
        return [pbu.get_pose(o, client=self.client) for o in self.objects]

    @property
    def client(self):
        return CLIENT_MAP[self.client_id]


@dataclass
class SceneState(State):
    world: World
    grasp: Optional[Grasp] = None

    @property
    def current_q(self):
        return pbu.get_joint_positions(
            self.world.robot,
            self.world.robot.get_group_joints(ARM_GROUP, client=self.world.client),
            client=self.world.client,
        )

    @property
    def poses(self):
        return self.world.poses

    def enforce_grasp(self):
        tool_link = pbu.link_from_name(
            self.world.robot, PANDA_TOOL_TIP, client=self.world.client
        )
        attachment = self.grasp.create_attachment(
            self.world.robot, link=tool_link, client=self.world.client
        )
        attachment.assign(client=self.world.client)

    def apply_sequence(
        self, sequence: List[Command], teleport=False, sim_steps=0, time_step=5e-3
    ):  # None | INF
        assert sequence is not None
        for command in sequence:
            if not isinstance(command, GroupTrajectory) or len(command.path) > 0:
                for _ in command.iterate(
                    self,
                    teleport=teleport,
                    time_step=time_step,
                    client=self.world.client,
                ):
                    # Derived values
                    if self.grasp is not None:
                        self.enforce_grasp()

                        for _ in range(sim_steps):
                            self.world.client.stepSimulation()

                    time.sleep(time_step)


def all_ycb_names() -> List[str]:
    return [ycb_type_from_file(path) for path in pbu.list_paths(YCB_PATH)]


def all_ycb_paths() -> List[str]:
    return pbu.list_paths(YCB_PATH)


def ycb_type_from_name(name: str) -> str:
    return name.split("_", 1)[-1]


def ycb_type_from_file(path: str) -> str:
    return ycb_type_from_name(os.path.basename(path))


def get_ycb_obj_path(ycb_type: str, use_concave=False) -> Optional[str]:
    path_from_type = {
        ycb_type_from_file(path): path
        for path in pbu.list_paths(YCB_PATH)
        if os.path.isdir(path)
    }

    if ycb_type not in path_from_type:
        return None

    if use_concave:
        filename = "google_16k/textured_vhacd.obj"
    else:
        filename = "google_16k/textured.obj"

    return os.path.join(path_from_type[ycb_type], filename)


def create_ycb(
    name,
    use_concave=True,
    client: Any = None,
    scale: float = 1.0,
    mass=-1,
    **kwargs,
) -> int:
    concave_ycb_path = get_ycb_obj_path(name, use_concave=use_concave)
    ycb_path = get_ycb_obj_path(name)

    color = pbu.WHITE
    mesh = trimesh.load(ycb_path)
    visual_geometry = pbu.get_mesh_geometry(ycb_path, scale=scale)
    collision_geometry = pbu.get_mesh_geometry(concave_ycb_path, scale=scale)
    geometry_pose = pbu.Pose(point=-mesh.center_mass)
    collision_id = pbu.create_collision_shape(
        collision_geometry, pose=geometry_pose, client=client
    )
    visual_id = pbu.create_visual_shape(
        visual_geometry, color=color, pose=geometry_pose, client=client
    )
    body = client.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
    )

    pbu.set_all_color(body, pbu.apply_alpha(color, alpha=1.0), client=client)

    return body


TABLE_AABB = pbu.AABB(
    lower=[-1.53 / 2.0, -1.22 / 2.0, -0.03 / 2.0],
    upper=[1.53 / 2.0, 1.22 / 2.0, 0.03 / 2.0],
)
TABLE_POSE = pbu.Pose((0.1, 0, -TABLE_AABB.upper[2]))


def create_default_env(
    client=None, table_aabb=TABLE_AABB, table_pose=TABLE_POSE, **kwargs
):
    client.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraYaw=90,
        cameraPitch=-15,
        cameraTargetPosition=[-0.5, 0, 0.3],
    )

    pbu.add_data_path()
    floor, _ = add_table(
        *pbu.get_aabb_extent(table_aabb), table_pose=table_pose, client=client
    )
    obstacles = [
        floor,  # collides with the robot when MAX_DISTANCE >= 5e-3
    ]

    for obst in obstacles:
        pbu.set_dynamics(
            obst,
            lateralFriction=1.0,  # linear (lateral) friction
            spinningFriction=1.0,  # torsional friction around the contact normal
            rollingFriction=0.01,  # torsional friction orthogonal to contact normal
            restitution=0.0,  # restitution: 0 => inelastic collision, 1 => elastic collision
            client=client,
        )

    return floor, obstacles


def add_table(
    table_width: float = 1.50,
    table_length: float = 1.22,
    table_thickness: float = 0.03,
    thickness_8020: float = 0.025,
    post_height: float = 1.25,
    color: Tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0),
    table_pose: pbu.PoseType = TABLE_POSE,
    client=None,
) -> Tuple[int, List[int]]:
    # Panda table downstairs very roughly (few cm of error)
    table = pbu.create_box(
        table_width, table_length, table_thickness, color=color, client=client
    )
    pbu.set_pose(table, table_pose, client=client)
    workspace = []

    # 80/20 posts and beams
    post_offset = thickness_8020 / 2  # offset (x, y) by half the thickness
    x_post = table_width / 2 - post_offset
    y_post = table_length / 2 - post_offset
    z_post = post_height / 2
    for mult_1 in [1, -1]:
        for mult_2 in [1, -1]:
            # Post
            post = pbu.create_box(
                thickness_8020, thickness_8020, post_height, color=color, client=client
            )
            pbu.set_pose(
                post,
                pbu.Pose((table_pose[0][0] + x_post * mult_1, y_post * mult_2, z_post)),
                client=client,
            )
            workspace.append(post)

    # 8020 cross-beams parallel in x-axis
    beam_offset = thickness_8020 / 2
    y_beam = table_length / 2 - beam_offset
    z_beam = post_height + beam_offset
    for mult in [1, -1]:
        beam = pbu.create_box(
            table_width, thickness_8020, thickness_8020, color=color, client=client
        )
        pbu.set_pose(
            beam, pbu.Pose((table_pose[0][0], y_beam * mult, z_beam)), client=client
        )
        workspace.append(beam)

    # 8020 cross-beams parallel in y-axis
    beam_length = table_length - 2 * thickness_8020
    x_beam = table_width / 2 - beam_offset
    for mult in [1, -1]:
        beam = pbu.create_box(
            thickness_8020, beam_length, thickness_8020, color=color, client=client
        )
        pbu.set_pose(
            beam, pbu.Pose((table_pose[0][0] + x_beam * mult, 0, z_beam)), client=client
        )
        workspace.append(beam)

    assert len(workspace) == 4 + 4  # 1 table, 4 posts, 4 beams
    return table, workspace


def pixel_from_point(camera_matrix, point_camera, width, height):
    px, py = pbu.pixel_from_ray(camera_matrix, point_camera)
    if (0 <= px < width) and (0 <= py < height):
        r, c = np.floor([py, px]).astype(int)
        return pbu.Pixel(r, c)
    return None


# A smaller version of the table only containing the region in front of the robot
def get_shortened_table_dims():
    TABLE_POSE = ((0.1 + 1.5 / 4, 0.0, -0.015), (0.0, 0.0, 0.0, 1.0))
    table_width = 0.75
    table_length = 1.22
    table_thickness = 0.015
    return TABLE_POSE, table_width, table_length, table_thickness


@contextlib.contextmanager
def suppress_c_output():
    # Redirect both stdout and stderr at the OS level
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)


def setup_robot_pybullet(gui=False):

    with suppress_c_output():
        client = bc.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        robot_body = pbu.load_pybullet(PANDA_PATH, fixed_base=True, client=client)
    return robot_body, client


def grasp_attachment(world: World, grasp: Grasp) -> pbu.Attachment:
    tool_link = pbu.link_from_name(world.robot, PANDA_TOOL_TIP, client=world.client)
    attachment = grasp.create_attachment(
        world.robot, link=tool_link, client=world.client
    )
    return attachment


def grasp_ik(world, obj, pose, grasp, obstacles=[]):
    pre_grasp_pose = pbu.multiply(pose, pbu.invert(grasp.pregrasp))
    grasp_pose = pbu.multiply(pose, pbu.invert(get_postgrasp(grasp.grasp)))

    pre_path = plan_workspace_motion(
        world.robot,
        [pre_grasp_pose, grasp_pose],
        obstacles=obstacles,
        client=world.client,
    )
    return pre_path


def ik(world, obj, pose, grasp):
    grasp_pose = pbu.multiply(pose, pbu.invert(grasp.grasp))
    arm_path = plan_workspace_motion(
        world.robot,
        [grasp_pose],
        obstacles=[world.floor],
        client=world.client,
    )
    if arm_path is None:
        return None
    else:
        return arm_path[0]


def dimensions_from_camera_image(camera_image: pbu.CameraImage):
    assert camera_image.rgbPixels.shape[:2] == camera_image.depthPixels.shape[:2]
    return camera_image.rgbPixels.shape[1], camera_image.rgbPixels.shape[0]


def plan_motion(world, q1, q2, attachments=[], obstacles=[]):
    conf1 = GroupConf(
        body=world.robot,
        group=ARM_GROUP,
        positions=q1,
        client=world.client,
    )

    conf2 = GroupConf(
        body=world.robot,
        group=ARM_GROUP,
        positions=q2,
        client=world.client,
    )

    if not np.allclose(conf1.positions, conf2.positions, 0.005):
        with pbu.LockRenderer(client=world.client):
            motion_plan = plan_motion_fn(
                world.robot,
                ARM_GROUP,
                conf1,
                conf2,
                attachments=attachments,
                environment=obstacles,
                client=world.client,
            )

            conf1.assign(client=world.client)

        return motion_plan
    return GroupTrajectory(world.robot, ARM_GROUP, [], client=world.client)


def pick_execute(state: SceneState, grasp, motion_plan, pre_grasp=[], full_close=False):
    teleport = not pbu.has_gui(client=state.world.client)
    # teleport=True
    state.apply_sequence([motion_plan], teleport=teleport)

    closed_position = 0 if full_close else grasp.closed_position
    close_conf = closed_position * np.ones(
        len(
            state.world.robot.get_group_joints(GRIPPER_GROUP, client=state.world.client)
        )
    )
    # Interpolate between current_gripper_pos and OPEN_GRIPPER_POS with a step size of 0.005.
    pos = np.linspace(
        OPEN_GRIPPER_POS,
        close_conf,
        int(np.linalg.norm(close_conf - OPEN_GRIPPER_POS) / GRIPPER_RES),
    )
    if len(pre_grasp) > 0:
        state.apply_sequence(
            [
                GroupTrajectory(
                    state.world.robot, ARM_GROUP, pre_grasp, client=state.world.client
                )
            ],
            teleport=teleport,
        )

    if len(pos) > 0:
        state.apply_sequence(
            [
                GroupTrajectory(
                    state.world.robot, GRIPPER_GROUP, pos, client=state.world.client
                )
            ],
            teleport=teleport,
        )
    state.grasp = grasp

    if len(pre_grasp) > 0:
        state.apply_sequence(
            [
                GroupTrajectory(
                    state.world.robot,
                    ARM_GROUP,
                    reversed(pre_grasp),
                    client=state.world.client,
                )
            ],
            teleport=teleport,
        )

    return state


def place_execute(state: SceneState, motion_plan, pre_grasp=[]):

    teleport = not pbu.has_gui(client=state.world.client)
    state.apply_sequence([motion_plan], teleport=teleport)
    state_grasp = state.grasp
    if state_grasp is not None:
        close_conf = state_grasp.closed_position * np.ones(
            len(
                state.world.robot.get_group_joints(
                    GRIPPER_GROUP, client=state.world.client
                )
            )
        )

        pos = np.linspace(
            close_conf,
            OPEN_GRIPPER_POS,
            int(np.linalg.norm(close_conf - OPEN_GRIPPER_POS) / GRIPPER_RES),
        )
    if len(pre_grasp) > 0:
        state.apply_sequence(
            [
                GroupTrajectory(
                    state.world.robot, ARM_GROUP, pre_grasp, client=state.world.client
                )
            ],
            teleport=teleport,
        )
    if len(pos) > 0:
        state.apply_sequence(
            [
                GroupTrajectory(
                    state.world.robot, GRIPPER_GROUP, pos, client=state.world.client
                )
            ],
            teleport=teleport,
        )

    state.grasp = None

    if len(pre_grasp) > 0:
        state.apply_sequence(
            [
                GroupTrajectory(
                    state.world.robot,
                    ARM_GROUP,
                    reversed(pre_grasp),
                    client=state.world.client,
                )
            ],
            teleport=teleport,
        )

    return state


def placement_sample(
    world: World,
    obj: int,
    region: int,
    surface_aabb: Optional[pbu.AABB] = None,
    surface_pose: Optional[pbu.PoseType] = None,
    **kwargs,
):

    # If surface_aabb and surface_pose are not provided, get them from the region
    if surface_aabb is None or surface_pose is None:
        surface_aabb = pbu.get_aabb(region, client=world.client)
        surface_pose = pbu.get_pose(region, client=world.client)

    assert surface_aabb is not None
    assert surface_pose is not None

    # Sample a random pose with a random theta on the surface that accounts for the object's size
    object_aabb = pbu.get_aabb(obj, client=world.client)
    object_size = np.array(object_aabb.upper) - np.array(object_aabb.lower)
    surface_size = np.array(surface_aabb.upper) - np.array(surface_aabb.lower)
    x = np.random.uniform(
        -surface_size[0] / 2 + object_size[0] / 2,
        surface_size[0] / 2 - object_size[0] / 2,
    )
    y = np.random.uniform(
        -surface_size[1] / 2 + object_size[1] / 2,
        surface_size[1] / 2 - object_size[1] / 2,
    )
    z = surface_size[2] / 2 + object_size[2] / 2 + 0.01
    theta = np.random.uniform(0, 2 * np.pi)
    placement = pbu.multiply(
        surface_pose, pbu.Pose(pbu.Point(x=x, y=y, z=z), pbu.Euler(0, 0, theta))
    )
    return placement
