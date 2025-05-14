from __future__ import annotations

import copy
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from tampura.config.config import register_env
from tampura.environment import TampuraEnv
from tampura.spec import ProblemSpec
from tampura.structs import (AbstractBelief, AbstractBeliefSet, Action,
                             ActionSchema, AliasStore, Belief, NoOp,
                             Observation, StreamSchema)
from tampura.symbolic import And, Atom, Exists, Not, Predicate

import tampura_environments.panda_utils.pb_utils as pbu
from tampura_environments.panda_utils.panda_env_utils import (
    ARM_GROUP, CLIENT_MAP, EXCLUDE_CLASSES, GRIPPER_GROUP, OPEN_GRIPPER_POS,
    SceneState, World, create_default_env, create_ycb,
    dimensions_from_camera_image, get_grasp, get_shortened_table_dims,
    grasp_attachment, grasp_ik, ik, pick_execute, pixel_from_point,
    place_execute, placement_sample, plan_motion, plan_workspace_motion,
    pose_to_vec, setup_robot, transformation_to_pose)
from tampura_environments.panda_utils.primitives import Grasp
from tampura_environments.panda_utils.robot import (CAMERA_FRAME,
                                                    DEFAULT_ARM_POS,
                                                    PANDA_TOOL_TIP)
from tampura_environments.panda_utils.voxel_utils import VoxelGrid

GRID_RESOLUTION = 0.015
SIM_CLIENT_ID = 2
TARGET_OBJECT = "dice"
GRASP_MODE = "saved"
EXECUTE_ATTEMPTS = 100


@dataclass
class SceneObservation(Observation):
    camera_image: Optional[pbu.CameraImage] = None
    poses: Dict[str, pbu.Pose] = field(default_factory=lambda: {})
    grasp: Optional[Grasp] = None
    grasp_body: Optional[str] = None
    conf: Optional[str] = None
    moved: Optional[str] = None
    possible_locations: List[Any] = field(default_factory=lambda: [])

    def __repr__(self):
        return f"SceneObservation(...)"


def obs_from_camera_image(
    object_syms: List[str],
    world: World,
    camera_image: pbu.CameraImage,
    conf,
    store: AliasStore,
) -> SceneObservation:
    seg_bodies = camera_image.segmentationMaskBuffer[:, :, 0]

    # Get the unique values of the first  (480, 640, 2) casted to int
    object_bodies = np.unique(seg_bodies.astype(int))
    object_poses = {}
    # For each object body, add it to known_objects if it is in obs.objects
    for body in object_bodies:
        for obj_sym, pose in zip(object_syms, world.poses):
            if store.get(obj_sym) == body:
                object_poses[obj_sym] = pose

    return SceneObservation(camera_image=camera_image, poses=object_poses, conf=conf)


class SceneBelief(Belief):
    def __init__(self, world: World, vis_grid=False, **kwargs):
        self.objects = {}
        self.camera_intrinsics = None
        self.world = world
        floor_aabb = pbu.get_aabb(self.world.floor, client=self.world.client)
        self.grasp = None
        self.grasp_body = None
        self.current_conf = None
        self.possible_locations = None
        self.placed = False
        visibility_aabb = pbu.AABB(
            lower=[
                0.37,
                floor_aabb.lower[1],
                floor_aabb.upper[2] + 0.02,
            ],
            upper=[
                floor_aabb.upper[0] - 0.05,
                floor_aabb.upper[1],
                floor_aabb.upper[2] + 0.1,
            ],
        )
        self.moved = []
        self.object_poses = {}
        self.vis_grid = vis_grid
        self.visibility_grid = self.setup_visibility_grid(visibility_aabb)

    def vectorize(self):
        obj_pose_vecs = []
        for obj in self.world.objects:
            if self.get_pose(obj) is None:
                obj_pose_vecs.append(pose_to_vec(pbu.unit_pose()))
            else:
                obj_pose_vecs.append(pose_to_vec(self.get_pose(obj)))
        vectorized_vis_grid = np.array(
            [self.visibility_grid.is_occupied(voxel) for voxel in self.get_all_voxels()]
        )
        b_vec = np.concatenate([vectorized_vis_grid] + obj_pose_vecs)
        return b_vec

    def get_pose(self, obj):
        if obj in self.object_poses:
            return self.object_poses[obj]
        else:
            return None

    def update_objects(self, obs: SceneObservation):
        """Given a camera image, use the image.segmentationMaskBuffer to update
        the known objects."""

        for obj_sym, pose in obs.poses.items():
            self.object_poses[obj_sym] = pose

    def get_all_voxels(self):
        return list(self.visibility_grid.voxels_from_aabb(self.visibility_grid.aabb))

    def set_sim(self, store: AliasStore):
        for obj_sym, pose in self.object_poses.items():
            if pose is None:
                pose = pbu.Pose(pbu.Point(x=100))  # Far away

            pbu.set_pose(store.get(obj_sym), pose, client=self.world.client)

    def get_new_seen_voxels(
        self, camera_image: pbu.CameraImage, include_unseen=False
    ) -> List[Any]:
        voxels = []
        width, height = dimensions_from_camera_image(camera_image)

        # For each voxel in the grid, check whether it was seen in the image
        for voxel in self.get_all_voxels():
            if self.visibility_grid.is_occupied(voxel):
                center_world = self.visibility_grid.to_world(
                    self.visibility_grid.center_from_voxel(voxel)
                )
                center_camera = pbu.tform_point(
                    pbu.invert(camera_image.camera_pose), center_world
                )
                distance = center_camera[2]
                pixel = pixel_from_point(
                    camera_image.camera_matrix, center_camera, width, height
                )
                if pixel is not None:
                    depth = camera_image.depthPixels[pixel.row, pixel.column]

                    if distance <= depth:
                        voxels.append(voxel)

                elif include_unseen:
                    voxels.append(voxel)

        return voxels

    def update_visibility(
        self,
        camera_image: pbu.CameraImage,
        store: AliasStore,
        possible_locations=None,
        include_unseen=False,
    ):
        """Updates the visibility grid based on a camera image.

        Args:
            camera_pose (tuple): Pose of the camera in the world frame.
            camera_image: The taken image from the camera.
            q (tuple): Robot configuration corresponding to the taken image.
        Returns:
            set: The gained vision obtaining from the given image.
        """
        new_voxels = self.get_new_seen_voxels(
            camera_image, include_unseen=include_unseen
        )
        for voxel in new_voxels:
            self.visibility_grid.set_free(voxel=voxel)

        # Remove points that are in collision with an object

        radius = 0.001
        point_collision_shape_id = self.world.client.createCollisionShape(
            p.GEOM_SPHERE, radius=radius
        )
        point_obj = self.world.client.createMultiBody(
            0, point_collision_shape_id, -1, [0, 0, 0]
        )
        for voxel in self.get_all_voxels():
            if self.visibility_grid.is_occupied(voxel):
                point = self.visibility_grid.center_from_voxel(voxel)

                if possible_locations is not None:
                    distances = []
                    for possible_point in possible_locations:
                        transformed_pose = transformation_to_pose(possible_point)[0]
                        transformed_array = np.array(transformed_pose)
                        point_array = np.array(point)
                        distance = np.linalg.norm(point_array - transformed_array)
                        distances.append(distance)

                    min_distance = min(float(distance) for distance in distances)

                    if min_distance > 0.05:
                        self.visibility_grid.set_free(voxel=voxel)
                        continue

                for obj_sym in self.object_poses:
                    pbu.set_pose(
                        point_obj,
                        pbu.Pose(point=pbu.Point(*point)),
                        client=self.world.client,
                    )

                    # Get the closest points
                    distance_threshold = (
                        GRID_RESOLUTION / 2.0
                    )  # Large enough to ensure the point is included
                    closest_points = self.world.client.getClosestPoints(
                        store.get(obj_sym), point_obj, distance_threshold
                    )
                    if len(closest_points) > 0:
                        self.visibility_grid.set_free(voxel=voxel)
        pbu.remove_body(point_obj, client=self.world.client)

    def setup_visibility_grid(self, surface: pbu.AABB) -> VoxelGrid:
        """Creates a grid that represents the visibility of the robot."""
        resolutions = GRID_RESOLUTION * np.ones(3)
        surface_origin = pbu.Pose(pbu.Point(z=0.01))
        surface_aabb = pbu.AABB(
            lower=surface.lower,
            upper=[surface.upper[0], surface.upper[1], GRID_RESOLUTION * 2],
        )

        # Defines two grids, one for visualization, and a second one for keeping track of regions during
        # planning.
        grid = VoxelGrid(
            resolutions,
            world_from_grid=surface_origin,
            aabb=surface_aabb,
            color=pbu.BLUE,
            client=self.world.client,
        )
        static_grid = VoxelGrid(
            resolutions,
            world_from_grid=surface_origin,
            aabb=surface_aabb,
            color=pbu.BLACK,
            client=self.world.client,
        )
        for voxel in grid.voxels_from_aabb(surface_aabb):
            grid.set_occupied(voxel)
            static_grid.set_occupied(voxel)

        return grid

    def update(
        self,
        action: Action,
        observation: SceneObservation,
        store: AliasStore,
    ) -> SceneBelief:
        new_belief = copy.deepcopy(self)

        if action.name == "place":
            new_belief.placed = True

        if observation.moved is not None:
            new_belief.moved = list(set(self.moved + [observation.moved]))

        new_belief.current_conf = observation.conf

        new_belief.update_objects(observation)
        new_belief.set_sim(store)

        if observation.possible_locations is not None:
            new_belief.possible_locations = observation.possible_locations

        if observation.camera_image is not None:
            new_belief.camera_intrinsics = observation.camera_image.camera_matrix
            new_belief.update_visibility(observation.camera_image, store)

        if observation.grasp is not None:
            new_belief.grasp = observation.grasp
            new_belief.grasp_body = observation.grasp_body
        else:
            new_belief.grasp = None
            new_belief.grasp_body = None

        return new_belief

    def abstract(self, store: AliasStore) -> AbstractBelief:
        ab = AbstractBelief([Atom("known-pose", [o]) for o in self.object_poses.keys()])

        if self.grasp is not None:
            ab.items.append(Atom("holding", [self.grasp_body]))
            ab.items.append(Atom("at-grasp", [self.grasp_body, self.grasp]))

        for obj, cat in zip(self.world.objects, self.world.categories):
            if cat == TARGET_OBJECT:
                ab.items.append(Atom("is-target", [obj]))

        ab.items += [Atom("moved", [p]) for p in self.moved]
        if self.current_conf is not None and np.allclose(
            self.current_conf, DEFAULT_ARM_POS, rtol=1e-4, atol=1e-4
        ):
            ab.items.append(Atom("at-home"))

        return ab


def pick_effects_fn(
    action: Action, belief: SceneBelief, store: AliasStore, **kwargs
) -> AbstractBeliefSet:

    # Our model of the pick action is that is succeeds if the grasp is valid and reachable.
    # We do not consider motion planning failures in our grasp model for this problem.

    # Get a grasp on the object
    (obj, grasp) = action.args
    g = store.get(grasp)
    belief.set_sim(store)
    new_belief = None

    if obj in belief.object_poses:
        pre_confs = grasp_ik(
            belief.world,
            store.get(obj),
            belief.get_pose(obj),
            g,
            obstacles=list(set(belief.world.environment) - set([store.get(obj)]))
            + [belief.world.floor],
        )
        if pre_confs is not None:
            new_belief = belief.update(
                action, SceneObservation(grasp=grasp, grasp_body=action.args[0]), store
            )

    if new_belief is None:
        return AbstractBeliefSet.from_beliefs([belief], store)

    return AbstractBeliefSet.from_beliefs([new_belief], store)


def pick_execute_fn(
    action: Action, belief: SceneBelief, state: SceneState, store: AliasStore
) -> Tuple[SceneState, SceneObservation]:
    (obj, grasp) = action.args
    g = store.get(grasp)

    obs = SceneObservation(conf=belief.current_conf)
    belief.set_sim(store)

    pre_confs = grasp_ik(
        belief.world,
        store.get(obj),
        belief.get_pose(obj),
        g,
        obstacles=list(set(belief.world.environment) - set([store.get(obj)]))
        + [belief.world.floor],
    )

    if pre_confs is None:
        logging.debug("Pick ik fail")
        return state, obs
    else:
        logging.debug("Pick ik success")

    motion_plan = plan_motion(
        belief.world,
        belief.current_conf,
        pre_confs[0],
        obstacles=[belief.world.floor] + list(set(belief.world.environment)),
    )
    if motion_plan is None:
        return state, obs

    state = pick_execute(state, g, motion_plan, pre_confs, full_close=False)
    obs.grasp = grasp
    obs.grasp_body = obj
    obs.conf = pre_confs[0]
    state.grasp = g
    return state, obs


def place_effects_fn(
    action: Action, belief: SceneBelief, store: AliasStore, **kwargs
) -> AbstractBeliefSet:

    # Our model of the place action is that is succeeds if the grasp is valid and reachable.

    (o, g, p) = action.args
    grasp = store.get(g)
    obj = store.get(o)
    placement_pose = store.get(p)
    belief.set_sim(store)
    new_belief = None

    pre_confs = grasp_ik(
        belief.world,
        obj,
        placement_pose,
        grasp,
        obstacles=list(set(belief.world.environment) - set([obj]))
        + [belief.world.floor],
    )
    if pre_confs is not None:
        new_belief = belief.update(
            action,
            SceneObservation(moved=o, poses={o: placement_pose}, conf=pre_confs),
            store,
        )

    if new_belief is None:
        return AbstractBeliefSet.from_beliefs([belief], store)

    return AbstractBeliefSet.from_beliefs([new_belief], store)


def place_execute_fn(
    action: Action, belief: SceneBelief, state: SceneState, store: AliasStore
) -> Tuple[SceneState, SceneObservation]:

    (obj, grasp, place_pose) = action.args
    p = store.get(place_pose)
    g = store.get(grasp)
    obs = SceneObservation(
        conf=belief.current_conf, grasp=g, grasp_body=belief.grasp_body
    )
    belief.set_sim(store)
    conf = ik(belief.world, store.get(obj), p, g)

    if conf is None:
        return state, obs

    motion_plan = plan_motion(
        belief.world,
        belief.current_conf,
        conf,
        obstacles=[belief.world.floor]
        + list(set(belief.world.environment) - set([store.get(obj)])),
        attachments=[grasp_attachment(belief.world, g)],
    )
    if motion_plan is None:
        return state, obs

    state = place_execute(state, motion_plan)
    obs.poses[obj] = p
    obs.conf = conf
    obs.grasp = None
    obs.grasp_body = None
    obs.moved = action.args[0]

    return state, obs


def look_effects_fn(
    action: Action, belief: SceneBelief, store: AliasStore, **kwargs
) -> AbstractBeliefSet:
    belief.set_sim(store)

    q = store.get(action.args[2])

    pbu.set_joint_positions(
        belief.world.robot,
        belief.world.robot.get_group_joints(ARM_GROUP, client=belief.world.client),
        q,
        client=belief.world.client,
    )

    camera_image = belief.world.robot.get_image(client=belief.world.client)

    new_seen_voxels = belief.get_new_seen_voxels(camera_image)
    new_seen_points = [
        belief.visibility_grid.center_from_voxel(voxel) for voxel in new_seen_voxels
    ]
    num_occupied = len(belief.visibility_grid.occupied)
    success_count = len(new_seen_points)
    fail_count = num_occupied - success_count
    new_fail_belief = belief.update(action, SceneObservation(camera_image), store)
    abstract_fail = new_fail_belief.abstract(store)

    belief_map = {
        abstract_fail: [new_fail_belief],
    }
    ab_counts = {abstract_fail: fail_count}

    if len(new_seen_points) > 0:
        new_success_belief = copy.deepcopy(new_fail_belief)
        new_success_belief.object_poses[action.args[0]] = pbu.Pose(
            pbu.Point(*random.choice(new_seen_points))
        )
        new_success_belief.current_conf = None
        abstract_success = new_success_belief.abstract(store)
        belief_map[abstract_success] = [new_success_belief]
        ab_counts[abstract_success] = success_count

        # new_success_belief.visibility_grid.draw_intervals(belief.world.client)
        # pbu.wait_if_gui(client=belief.world.client)

    abstract_bs = AbstractBeliefSet(
        ab_counts=ab_counts,
        belief_map=belief_map,
    )

    return abstract_bs


def look_execute_fn(
    action: Action, belief: SceneBelief, state: SceneState, store: AliasStore
) -> Tuple[SceneState, SceneObservation]:
    observation = SceneObservation(conf=belief.current_conf)
    belief.set_sim(store)
    look_conf = store.get(action.args[2])
    motion_plan = plan_motion(
        belief.world,
        belief.current_conf,
        look_conf,
        obstacles=[belief.world.floor] + belief.world.environment,
    )
    if motion_plan is None:
        return state, observation

    if len(motion_plan.path) > 0:
        state.apply_sequence([motion_plan])

    camera_image = state.world.robot.get_image(client=state.world.client)
    obs = obs_from_camera_image(
        belief.world.objects, state.world, camera_image, look_conf, store
    )
    obs.conf = look_conf

    return state, obs


def home_effects_fn(
    action: Action, belief: SceneBelief, store: AliasStore
) -> AbstractBeliefSet:
    new_belief = belief.update(
        action,
        SceneObservation(
            grasp=belief.grasp, grasp_body=belief.grasp_body, conf=DEFAULT_ARM_POS
        ),
        store=store,
    )
    return AbstractBeliefSet.from_beliefs([new_belief], store)


def home_execute_fn(
    action: Action, belief: SceneBelief, state: SceneState, store: AliasStore
) -> Tuple[SceneState, SceneObservation]:
    observation = SceneObservation(
        grasp=belief.grasp, grasp_body=belief.grasp_body, conf=DEFAULT_ARM_POS
    )
    belief.set_sim(store)
    motion_plan = plan_motion(
        belief.world,
        belief.current_conf,
        DEFAULT_ARM_POS,
        obstacles=[belief.world.floor]
        + list(set(belief.world.environment) - {store.get(belief.grasp_body)}),
        attachments=[grasp_attachment(belief.world, store.get(belief.grasp))],
    )
    if motion_plan is None:
        return state, observation
    state.apply_sequence([motion_plan])
    return state, observation


def look_sample_fn_wrapper(b0, max_attempts=100):
    def look_sample_fn(args: List[str], store: AliasStore):
        """We want the configuration of a random camera pose that's pointed
        roughly at the table.

        Out of laziness and not wanting to think about frame math right
        now, generate and test!
        """
        (obj_sym,) = args
        look_at_obj = store.get(obj_sym)

        def look_at(camera_position, target_position, up_direction=np.array([0, 0, 1])):
            # Calculate the forward vector from the camera to the target
            forward = np.array(target_position) - np.array(camera_position)
            forward = forward / np.linalg.norm(forward)

            # Calculate the left vector
            left = np.cross(up_direction, forward)
            left = left / np.linalg.norm(left)

            # Recalculate the orthonormal up vector
            up = np.cross(forward, left)
            up = up / np.linalg.norm(up)

            # Construct the rotation matrix
            rotation_matrix = np.array([left, up, forward]).T
            return rotation_matrix

        num_attempts = 0
        while num_attempts < max_attempts:
            num_attempts += 1
            client = b0.world.client
            table_aabb = pbu.get_aabb(b0.world.floor, client=client)
            object_pose = pbu.get_pose(look_at_obj, client=client)

            target_world_T_camera = pbu.multiply(
                pbu.Pose(
                    pbu.Point(
                        x=np.random.normal(0, 0.1),
                        y=np.random.normal(0, 0.1),
                        z=np.random.uniform(0.2, 0.3),
                    )
                ),
                object_pose,
            )
            r = look_at(target_world_T_camera[0], object_pose[0])
            rotation = R.from_matrix(r)
            quaternion = rotation.as_quat()

            target_world_T_camera = pbu.multiply(
                pbu.Pose(target_world_T_camera[0]), ([0, 0, 0], quaternion)
            )
            target_world_T_gripper = target_world_T_camera
            current_world_T_camera = pbu.get_link_pose(
                b0.world.robot,
                pbu.link_from_name(b0.world.robot, CAMERA_FRAME, client=client),
                client=client,
            )
            current_world_T_gripper = pbu.get_link_pose(
                b0.world.robot,
                pbu.link_from_name(b0.world.robot, PANDA_TOOL_TIP, client=client),
                client=client,
            )
            camera_T_gripper = pbu.multiply(
                pbu.invert(current_world_T_camera), current_world_T_gripper
            )
            target_world_T_gripper = pbu.multiply(
                target_world_T_camera, camera_T_gripper
            )

            arm_path = plan_workspace_motion(
                b0.world.robot,
                [target_world_T_gripper],
                obstacles=[b0.world.floor],
                client=b0.world.client,
            )
            if arm_path is None:
                continue
            else:
                pbu.set_joint_positions(
                    b0.world.robot,
                    b0.world.robot.get_group_joints(ARM_GROUP, client=b0.world.client),
                    arm_path[0],
                    client=b0.world.client,
                )
                points = []
                for x in np.linspace(0.2, table_aabb.upper[0], num=10):
                    for y in np.linspace(
                        table_aabb.lower[1], table_aabb.upper[1], num=20
                    ):
                        points.append(pbu.Point(x, y))

                return arm_path[0]

        return None

    return look_sample_fn


def setup_world(robot, client, categories, poses, **kwargs) -> World:
    """Setup the world to have a set of YCB objects at certain poses defined in
    the scene_data dictionary."""

    floor, obstacles = create_default_env(client=client, **kwargs)

    movable = []
    for category in categories:
        obj = create_ycb(category, client=client, use_concave=True)
        logging.debug(f"Created object {category} with body {obj}")
        movable.append(obj)

    client_id = len(CLIENT_MAP)
    CLIENT_MAP[client_id] = client

    for obj, pose in zip(movable, poses):
        pbu.set_pose(obj, pose, client=client)

    pbu.set_joint_positions(
        robot,
        robot.get_group_joints(ARM_GROUP, client=client),
        DEFAULT_ARM_POS,
        client=client,
    )

    pbu.set_joint_positions(
        robot,
        robot.get_group_joints(GRIPPER_GROUP, client=client),
        OPEN_GRIPPER_POS,
        client=client,
    )

    return World(
        client_id=client_id,
        robot=robot,
        environment=obstacles + movable,
        floor=floor,
        objects=movable,
        categories=categories,
    )


def upsample_transformations(matrices, K):
    N = matrices.shape[0]
    if K <= N:
        return matrices

    # Function to apply perturbation and randomize yaw
    def modify_matrix(matrix, dx, dy):
        modified = np.copy(matrix)
        modified[0, 3] += dx  # perturb x
        modified[1, 3] += dy  # perturb y

        # Randomize yaw (rotation around z-axis)
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        modified[:3, :3] = np.dot(rotation_matrix, modified[:3, :3])
        return modified

    # Upsample the matrix list
    output = []
    for matrix in matrices:
        output.append(matrix)
        for _ in range((K - N) // N):
            dx, dy = (
                np.random.randn(2) * 0.05
            )  # Example perturbation values for x and y
            output.append(modify_matrix(matrix, dx, dy))

    return np.array(output)[:K]


def grasp_sample_fn_wrapper(world):
    def grasp_sample_fn(args: List[str], store: AliasStore):
        (obj,) = store.get_all(args)
        grasp = get_grasp(
            world,
            obj,
            world.environment,
            grasp_mode=GRASP_MODE,
            use_saved=False,
            client=world.client,
        )
        return grasp

    return grasp_sample_fn


def placement_sample_fn_wrapper(world):
    def place_sample_fn(args: List[str], store: AliasStore):
        (obj,) = store.get_all(args)

        table_pose, table_width, table_length, thickness = get_shortened_table_dims()
        table_aabb = pbu.AABB(
            lower=[-table_width / 2, -table_length / 2, -thickness / 2],
            upper=[table_width / 2, table_length / 2, thickness / 2],
        )

        placement = placement_sample(
            world,
            obj,
            world.floor,
            world.environment,
            table_aabb=table_aabb,
            table_pose=table_pose,
            client=world.client,
        )
        return placement

    return place_sample_fn


class FindDiceEnv(TampuraEnv):
    def __init__(self, *args, **kwargs):
        super(FindDiceEnv, self).__init__(*args, **kwargs)
        self.world = None

    def get_scene_data(self):
        dataset_dir = "tampura_environments/find_dice/problems"
        world_json_file = random.choice(os.listdir(dataset_dir))

        logging.info("Loading scene from {}".format(world_json_file))

        # Load json from file
        with open(os.path.join(dataset_dir, world_json_file)) as f:
            scene_data_json = json.load(f)

        return scene_data_json

    def vis_updated_belief(self, belief: SceneBelief, store: AliasStore):
        belief.visibility_grid.draw_intervals(self.world.client)

    def initialize(self) -> Tuple[SceneBelief, AliasStore]:
        store = AliasStore()

        state_vis = self.vis
        belief_vis = False

        world_robot, world_client = setup_robot(vis=state_vis)
        sim_world_robot, sim_world_client = setup_robot(
            vis=belief_vis,
            camera_matrix=world_robot.camera.camera_matrix,
        )
        possible_locations = None
        scene_data = None
        while scene_data is None:
            scene_data_json = self.get_scene_data()
            if set(EXCLUDE_CLASSES).isdisjoint(set(scene_data_json["categories"])):
                scene_data = scene_data_json
                self.world = setup_world(
                    world_robot,
                    world_client,
                    scene_data["categories"],
                    scene_data["poses"],
                    vis=state_vis,
                )

        self.state = SceneState(self.world)
        self.sim_world = setup_world(
            sim_world_robot,
            sim_world_client,
            scene_data["categories"],
            scene_data["poses"],
            vis=belief_vis,
        )
        self.sim_world.objects = [
            store.add_typed(o, "physical") for o in self.sim_world.objects
        ]

        camera_image = self.world.robot.get_image(client=self.world.client)

        obs = obs_from_camera_image(
            self.sim_world.objects, self.world, camera_image, DEFAULT_ARM_POS, store
        )

        # Construct an initial belief from the observation
        b = SceneBelief(self.sim_world, vis_grid=True)
        b.update_visibility(
            camera_image,
            store,
            possible_locations=possible_locations,
            include_unseen=True,
        )
        b.visibility_grid.draw_intervals(b.world.client)
        b = b.update(NoOp(), obs, store=store)
        b.set_sim(store)
        self.starting_belief = b

        # Visualize the visibility grid in pybullet
        if self.vis:
            b.visibility_grid.draw_intervals(self.world.client)

        return b, store

    def get_problem_spec(self):
        predicates = [
            Predicate("known-pose", ["physical"]),
            Predicate("holding", ["physical"]),
            Predicate("at-start", ["physical"]),
            Predicate("object-pose", ["physical", "pose"]),
            # Predicate("at-pose", ["physical", "pose"]),
            Predicate("is-target", ["physical"]),
            Predicate("at-home", []),
            Predicate("looking-conf", ["physical", "conf"]),
            Predicate("moved", ["physical"]),
            Predicate("object-grasp", ["physical", "grasp"]),
            Predicate("at-grasp", ["physical", "grasp"]),
        ]

        stream_schemas = [
            StreamSchema(
                name="look-conf-sample",
                inputs=["?o"],
                input_types=["physical"],
                output="?q",
                output_type="conf",
                certified=[Atom("looking-conf", ["?o", "?q"])],
                sample_fn=look_sample_fn_wrapper(self.starting_belief),
            ),
            StreamSchema(
                name="grasp-sample",
                inputs=["?o"],
                input_types=["physical"],
                output="?g",
                output_type="grasp",
                certified=[Atom("object-grasp", ["?o", "?g"])],
                sample_fn=grasp_sample_fn_wrapper(self.sim_world),
            ),
            StreamSchema(
                name="place-sample",
                inputs=["?o"],
                input_types=["physical"],
                output="?p",
                output_type="pose",
                certified=[Atom("object-pose", ["?o", "?p"])],
                sample_fn=placement_sample_fn_wrapper(self.sim_world),
            ),
        ]

        holding = Exists(Atom("holding", ["?obj"]), ["?obj"], ["physical"])
        action_schemas = [
            ActionSchema(
                name="pick",
                inputs=["?o", "?g"],
                input_types=["physical", "grasp"],
                preconditions=[
                    Not(Atom("moved", ["?o"])),
                    Not(holding),
                    Atom("known-pose", ["?o"]),
                    Atom("object-grasp", ["?o", "?g"]),
                ],
                effects_fn=pick_effects_fn,
                execute_fn=pick_execute_fn,
                effects=[Not(Atom("at-home"))],
                verify_effects=[
                    Atom("holding", ["?o"]),
                    Atom("at-grasp", ["?o", "?g"]),
                ],
            ),
            ActionSchema(
                name="place",
                inputs=["?o", "?g", "?p"],
                input_types=["physical", "grasp", "pose"],
                preconditions=[
                    Atom("holding", ["?o"]),
                    Atom("at-grasp", ["?o", "?g"]),
                    Atom("object-pose", ["?o", "?p"]),
                ],
                effects_fn=place_effects_fn,
                execute_fn=place_execute_fn,
                effects=[
                    Atom("moved", ["?o"]),
                    Not(Atom("at-home")),
                    Not(Atom("holding", ["?o"])),
                    Not(Atom("at-grasp", ["?o", "?g"])),
                    Atom("known-pose", ["?o"]),
                ],
            ),
            ActionSchema(
                name="look",
                inputs=["?o1", "?o2", "?q"],  # looking behind ?o2 for object ?o1
                input_types=["physical", "physical", "conf"],
                depends=[Atom("moved", ["?o2"])],
                preconditions=[
                    Atom("looking-conf", ["?o2", "?q"]),
                    Not(Atom("known-pose", ["?o1"])),
                    Atom("known-pose", ["?o2"]),
                    Not(holding),
                ],
                effects_fn=look_effects_fn,
                execute_fn=look_execute_fn,
                effects=[Not(Atom("at-home"))],
                verify_effects=[Atom("known-pose", ["?o1"])],
            ),
            ActionSchema(
                name="go-home",
                inputs=[],
                input_types=[],
                preconditions=[
                    holding,
                    Not(Atom("at-home")),
                ],
                effects_fn=home_effects_fn,
                execute_fn=home_execute_fn,
                effects=[Atom("at-home")],
            ),
            NoOp(),
        ]

        holding_die = Exists(
            And(
                [
                    Atom("is-target", ["?o"]),
                    Atom("holding", ["?o"]),
                ]
            ),
            ["?o"],
            ["physical"],
        )

        reward = And([holding_die, Atom("at-home")])

        spec = ProblemSpec(
            predicates=predicates,
            action_schemas=action_schemas,
            stream_schemas=stream_schemas,
            reward=reward,
        )

        return spec


class FindDiceEnvSimple(FindDiceEnv):
    """Only contains a single die in a reachable area."""

    def get_scene_data(self):
        scene_data_json = {
            "categories": ["dice"],
            "poses": [
                [
                    [0.6256510695393028, 0.0, 0.02085035479050672],
                    [0.0, 0.0, -0.9148730176578315, 0.4037417015390575],
                ]
            ],
        }
        return scene_data_json


register_env("find_dice", FindDiceEnv)
register_env("find_dice_simple", FindDiceEnvSimple)
