from __future__ import annotations

import copy
import logging
import math
import os
import random
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from math import atan2, pi
from pathlib import Path
from typing import Any, List, Tuple

import imageio.v3 as iio  # pip install imageio pillow
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
from tampura.config.config import register_env
from tampura.environment import TampuraEnv
from tampura.spec import ProblemSpec
from tampura.structs import (AbstractBelief, AbstractBeliefSet, Action,
                             ActionSchema, AliasStore, Belief, NoOp,
                             Observation, State)
from tampura.symbolic import (OBJ, And, Atom, Eq, Exists, ForAll, Imply, Not,
                              Predicate, When)


@dataclass
class Size:
    width: float
    height: float

    def __hash__(self):
        return hash((self.width, self.height))

    def to_array(self):
        return np.array([self.width, self.height])


TARGET_SHAPE = Size(0.5, 0.5)
BEACON_RADIUS = 0.2
BEACON_SHAPE = Size(BEACON_RADIUS * 2, BEACON_RADIUS * 2)
VISION_RADIUS = BEACON_RADIUS * 5
VISION_SHAPE = Size(VISION_RADIUS * 2, VISION_RADIUS * 2)
ROBOT_SHAPE = Size(0.3, 0.6)
GOAL_RADIUS = 0.4
ENV_SIZE = 10
ACTION_STD = 0.04
# ACTION_STD = 0.01 # Standard
# ACTION_STD = 0.0
KNOWN_VAR_THRESH = 0.05
MAX_ROTATION = pi / 6.0
STEP_SIZE = 0.2
REGION_IN_THRESH = 0.8


@dataclass
class SlamObservation(Observation):
    regions_in: List[str] = field(default_factory=lambda: [])
    holding: str = None
    initial_state: SlamState = None
    primitives: List[Any] = field(default_factory=lambda: [])
    robot_pose: Pose = None
    collision: bool = False

    def __str__(self):
        out = "In regions: {}, holding: {}, robot pose: {}".format(
            str(", ".join([r for r in self.regions_in])), self.holding, self.robot_pose
        )
        return out

    def __hash__(self):
        return hash(
            tuple(
                self.regions_in
                + [self.holding is None]
                + [self.robot_pose]
                + [self.collision]
            )
        )

    def __eq__(self, o2):
        return hash(self) == hash(o2)


class SlamBelief(Belief):
    def __init__(self, obs, num_particles, store, **kwargs):
        self.num_particles = num_particles
        assert (
            obs.initial_state is not None
        ), "Robot pose must be known in the initial state"
        self.particles = [
            copy.deepcopy(obs.initial_state) for _ in range(num_particles)
        ]

    def particle_replenish(self, particles):
        if len(particles) == 0:
            return []

        current_size = len(particles)

        if current_size >= self.num_particles:
            return particles

        replenish_size = self.num_particles - current_size

        for _ in range(replenish_size):
            random_particle = random.choice(particles)
            new_particle = copy.deepcopy(random_particle)
            particles.append(new_particle)

        return particles

    def vectorize(self):
        return np.concatenate([state.to_array() for state in self.particles])

    def update(
        self,
        action: Action,
        observation: SlamObservation,
        store: AliasStore,
    ) -> Belief:
        new_belief = copy.deepcopy(self)

        if observation is None:
            return new_belief

        if len(observation.primitives) > 0:
            trajectories, observations = generate_trajectory(
                self.particles, observation.primitives, std=ACTION_STD
            )
            new_belief.particles = [t[-1] for t in trajectories]
            for p, o in zip(new_belief.particles, observations):
                if o.collision:
                    p.in_collision = True

        equals_obs = []
        region_poses = np.array(
            [
                [region.pose.to_array() for region in p.regions]
                for p in new_belief.particles
            ]
        )
        region_shapes = np.array(
            [
                [region.shape.to_array() for region in p.regions]
                for p in new_belief.particles
            ]
        )
        target_poses = np.array(
            [
                [target.pose.to_array() for target in p.targets]
                for p in new_belief.particles
            ]
        )
        target_shapes = np.array(
            [
                [target.shape.to_array() for target in p.targets]
                for p in new_belief.particles
            ]
        )

        for pi, particle in enumerate(new_belief.particles):
            ins = []
            targets_in = []
            collisions = np.squeeze(
                collides(
                    unsqueeze(particle.robot.pose.to_array()),
                    unsqueeze(ROBOT_SHAPE.to_array()),
                    region_poses[pi, :],
                    region_shapes[pi, :],
                )
            )
            (collision_idx,) = np.where(collisions == 1)
            for cidx in collision_idx:
                ins.append(particle.regions[cidx].name)

            if target_poses[pi, :].shape[0] > 0:
                collisions = np.squeeze(
                    collides(
                        unsqueeze(particle.robot.pose.to_array()),
                        unsqueeze(ROBOT_SHAPE.to_array()),
                        target_poses[pi, :],
                        target_shapes[pi, :],
                    )
                )

                (collision_idx,) = np.where(collisions == 1)
                for cidx in collision_idx:
                    targets_in.append(particle.targets[cidx].name)

            if set(ins) == set(observation.regions_in):
                if observation.holding is None:
                    particle.attachments = []
                    equals_obs.append(particle)
                else:
                    if observation.holding.target in targets_in:
                        particle.attachments = [observation.holding]
                        equals_obs.append(particle)

        if observation.robot_pose is not None:
            for particle in new_belief.particles:
                particle.robot.pose = observation.robot_pose

        if len(equals_obs) > 0:
            new_belief.particles = self.particle_replenish(equals_obs)

        return new_belief

    def abstract(self, store: AliasStore) -> AbstractBelief:
        known = []
        v_x = np.std([p.robot.pose.x for p in self.particles]) ** 2
        v_y = np.std([p.robot.pose.y for p in self.particles]) ** 2

        if v_x < KNOWN_VAR_THRESH and v_y < KNOWN_VAR_THRESH:
            known.append(Atom("known_pose"))

        regions_in = defaultdict(lambda: 0)
        targets_in = defaultdict(lambda: defaultdict(lambda: 0))
        region_poses = np.array(
            [[region.pose.to_array() for region in p.regions] for p in self.particles]
        )
        region_shapes = np.array(
            [[region.shape.to_array() for region in p.regions] for p in self.particles]
        )
        target_poses = np.array(
            [[target.pose.to_array() for target in p.targets] for p in self.particles]
        )
        target_shapes = np.array(
            [[target.shape.to_array() for target in p.targets] for p in self.particles]
        )

        for pi, particle in enumerate(self.particles):
            collisions = np.squeeze(
                collides(
                    unsqueeze(particle.robot.pose.to_array()),
                    unsqueeze(ROBOT_SHAPE.to_array()),
                    region_poses[pi, :],
                    region_shapes[pi, :],
                )
            )
            (collision_idx,) = np.where(collisions == 1)
            for cidx in collision_idx:
                regions_in[particle.regions[cidx].name] += 1 / float(
                    len(self.particles)
                )

            if region_poses[pi, :].shape[0] > 0 and target_poses[pi, :].shape[0] > 0:
                collisions = collides(
                    region_poses[pi, :],
                    region_shapes[pi, :],
                    target_poses[pi, :],
                    target_shapes[pi, :],
                )

                c_region, c_target = np.where(collisions == 1)
                for cr, ct in zip(c_region, c_target):
                    targets_in[particle.regions[cr].name][
                        particle.targets[ct].name
                    ] += 1 / float(len(self.particles))

        for region_name, prob in regions_in.items():
            if prob > REGION_IN_THRESH:
                known.append(Atom("in", [region_name]))

        for region_name in store.type_dict["region"]:
            for target_name, prob in targets_in[region_name].items():
                if prob > REGION_IN_THRESH:
                    known.append(Atom("target_in", [target_name, region_name]))

        if all([len(p.attachments) > 0 for p in self.particles]):
            known.append(Atom("holding", [self.particles[0].attachments[0].target]))

        if any([p.in_collision for p in self.particles]):
            known.append(Atom("in_collision"))

        return AbstractBelief(known)


def effects_wrapper(get_trajectory):
    def effects_fn(
        action: Action, belief: SlamBelief, store: AliasStore
    ) -> AbstractBeliefSet:
        actions = get_trajectory(action, belief, store)

        if actions is None:
            actions = []

        trajectories, observations = generate_trajectory(
            belief.particles, actions, std=ACTION_STD
        )
        new_states = [t[-1] for t in trajectories]

        new_belief = copy.deepcopy(belief)
        for particle, new_p, obs in zip(new_belief.particles, new_states, observations):
            particle.robot.pose = copy.deepcopy(new_p.robot.pose)
            particle.in_collision = obs.collision
            for target, new_p_target in zip(particle.targets, new_p.targets):
                target.pose = new_p_target.pose

        if action.name == "move_look" or action.name == "move_corner":
            for obs, new_state in zip(observations, new_states):
                if action.args[0] in obs.regions_in:
                    obs.robot_pose = new_state.robot.pose

        # Avoid redundantly belief updating for identical observations
        hash_map = {}
        count_map = defaultdict(lambda: 0)
        belief_map = {}
        for obs in observations:
            if hash(obs) in hash_map:
                count_map[hash_map[hash(obs)]] += 1
                continue

            obs.primitives = []
            new_belief_updated = new_belief.update(action, obs, store)
            ab = new_belief_updated.abstract(store)
            belief_map[ab] = [new_belief_updated]
            count_map[ab] += 1
            hash_map[hash(obs)] = ab

        ab_set = AbstractBeliefSet(count_map, belief_map)

        return ab_set

    return effects_fn


def execute_wrapper(get_trajectory, save_dir, vis):
    def execute_fn(
        action: Action, belief: SlamBelief, state: SlamState, store: AliasStore
    ) -> Tuple[SlamState, SlamObservation]:
        actions = get_trajectory(action, belief, store)

        if actions is None:
            actions = []

        trajectories, observations = generate_trajectory(
            [state], actions, std=ACTION_STD
        )

        if vis:
            b_trajectories, b_observations = generate_trajectory(
                belief.particles, actions, std=ACTION_STD, vis=True
            )

            # import pickle
            # import time

            # filename = str(time.time())+str(".pkl")
            # filehandler = open(filename, 'wb')
            # pickle.dump({"trajectories": b_trajectories,
            #              "observations": b_observations,
            #              "state": state,
            #              "action": action,
            #              "primitives": actions,
            #              "belief": belief,
            #              "store": store}, filehandler)
            plot_trajectories(b_trajectories, save_dir)

        next_state = trajectories[0][-1]

        if action.name == "move_look" or action.name == "move_corner":
            if action.args[0] in observations[0].regions_in:
                observations[0].robot_pose = next_state.robot.pose

        return next_state, observations[0]

    return execute_fn


def clamp(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, ENV_SIZE)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float

    def to_array(self):
        return np.array([self.x, self.y, self.theta])

    @classmethod
    def from_array(cls, array):
        assert array.shape[0] == 3
        return Pose(array[0], array[1], array[2])


def multiply_poses(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Extracting x, y, theta from a
    a_x, a_y, a_theta = a[:, 0], a[:, 1], a[:, 2]

    # Extracting x, y, theta from b
    b_x, b_y, b_theta = b[:, 0], b[:, 1], b[:, 2]

    # Compute new x, y, and theta values using the same math as in the original function
    new_x = a_x + b_x * np.cos(a_theta) - b_y * np.sin(a_theta)
    new_y = a_y + b_x * np.sin(a_theta) + b_y * np.cos(a_theta)
    new_theta = a_theta + b_theta

    # Stack the new x, y, and theta values into a Nx3 array
    result = np.stack([new_x, new_y, new_theta], axis=1)

    return result


def inverse(a: np.ndarray) -> np.ndarray:
    # Extracting x, y, theta from a
    a_x, a_y, a_theta = a[:, 0], a[:, 1], a[:, 2]

    # Compute inverse theta
    inv_theta = -a_theta

    # Compute inverse x and y using the same math as in the original function
    inv_x = -a_x * np.cos(inv_theta) + a_y * np.sin(inv_theta)
    inv_y = -a_x * np.sin(inv_theta) - a_y * np.cos(inv_theta)

    # Stack the inv_x, inv_y, and inv_theta values into a Nx3 array
    result = np.stack([inv_x, inv_y, inv_theta], axis=1)

    return result


@dataclass
class SlamAction:
    delta_x: float
    delta_y: float
    attach: bool = False
    detach: bool = False


@dataclass
class Attachment:
    target: str
    robot_T_target: Pose


@dataclass
class SlamState(State):
    robot: Region
    obstacles: List[Region] = field(default_factory=lambda: [])
    beacons: List[Region] = field(default_factory=lambda: [])
    targets: List[Region] = field(default_factory=lambda: [])
    corners: List[Region] = field(default_factory=lambda: [])
    attachments: List[Attachment] = field(default_factory=lambda: [])
    goal: str = None
    start: str = None
    in_collision: bool = False

    @property
    def regions(self):
        return self.beacons + self.corners + [self.goal] + [self.start]

    @property
    def subtypes(self):
        return (
            ["beacon"] * len(self.beacons)
            + ["corner"] * len(self.corners)
            + ["goal", "start"]
        )

    def to_array(self):
        """Convert dynamic aspects of slam object to numpy array for vectorized
        operations."""
        target_names = [t.name for t in self.targets]
        target_poses = [t.pose.to_array() for t in self.targets]
        target_offsets = [Pose(0, 0, 0).to_array() for _ in range(len(self.targets))]
        for attachment in self.attachments:
            target_offsets[target_names.index(attachment.target)] = (
                attachment.robot_T_target.to_array()
            )

        cat = [self.robot.pose.to_array()] + target_poses + target_offsets
        return np.concatenate(cat)

    def from_array(self, array: np.ndarray):
        # TODO: make class method instead of deepcopying a reference
        ref_copy = copy.deepcopy(self)
        ref_copy.robot.pose = Pose.from_array(array[:3])
        for ti in range(len(ref_copy.targets)):
            ref_copy.targets[ti].pose = Pose.from_array(array[3 + 3 * ti : 6 + 3 * ti])
        return ref_copy


def sample_point_within_region(center_pose: Pose, region_size: Size) -> Pose:
    """Sample a random point within a rectangular region.

    Parameters:
        - center_pose: Pose object representing the center of the rectangular region
        - region_size: Size object representing the width and height of the rectangular region

    Returns:
        - A Pose object representing a point within the region
    """
    # Randomly sample a point within the rectangle centered around center_pose
    x = random.uniform(
        center_pose.x - region_size.width / 2, center_pose.x + region_size.width / 2
    )
    y = random.uniform(
        center_pose.y - region_size.height / 2, center_pose.y + region_size.height / 2
    )

    # For the angle, you can either keep the orientation of the center pose or sample a new one
    theta = center_pose.theta  # Here I keep the orientation of the center pose

    return Pose(x, y, theta)


@dataclass
class Region:
    pose: Pose = field(default_factory=lambda: Pose(0.0, 0.0, 0.0))
    shape: Size = field(default_factory=lambda: Size(1.0, 1.0))
    exact: bool = False
    name: str = "unknown"

    def __hash__(self):
        return hash((self.pose, self.shape))

    def sample(self):
        pose = self.pose  # sample_point_within_region(self.pose, self.shape)
        return Region(pose, self.shape)


def get_corners(pose, size):
    """Given a pose array of shape (N, 3) and a size array of shape (N, 2),
    return an array of corners of size (N, 4, 2)"""

    if pose.shape[0] == 0:
        return []

    N = pose.shape[0]
    half_widths, half_heights = size[:, 0] / 2, size[:, 1] / 2

    x_offsets = np.array([1, -1, -1, 1]).reshape(1, 4)  # shape (1, 4)
    y_offsets = np.array([1, 1, -1, -1]).reshape(1, 4)  # shape (1, 4)

    x_corners = half_widths.reshape(N, 1) * x_offsets  # shape (N, 4)
    y_corners = half_heights.reshape(N, 1) * y_offsets  # shape (N, 4)

    corners = np.stack([x_corners, y_corners], axis=2)  # shape (N, 4, 2)

    theta = pose[:, 2].reshape(N, 1)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix = np.empty((N, 2, 2))
    rotation_matrix[:, 0, 0] = cos_theta[:, 0]
    rotation_matrix[:, 0, 1] = -sin_theta[:, 0]
    rotation_matrix[:, 1, 0] = sin_theta[:, 0]
    rotation_matrix[:, 1, 1] = cos_theta[:, 0]
    rotated_corners = np.matmul(rotation_matrix, corners.transpose(0, 2, 1)).transpose(
        0, 2, 1
    )
    translated_corners = rotated_corners + pose[:, :2].reshape(N, 1, 2)
    return translated_corners


def check_corners_inside_rects(corners, rects):
    corners_exp = corners[:, :, np.newaxis, :]
    rects_exp = rects[np.newaxis, :, :, :]

    AB = rects_exp[:, :, 1, :] - rects_exp[:, :, 0, :]
    AM = corners_exp - rects_exp[:, :, 0, :]
    BC = rects_exp[:, :, 2, :] - rects_exp[:, :, 1, :]
    BM = corners_exp - rects_exp[:, :, 1, :]

    # Adjust the shape of AB and BC for correct broadcasting
    AB_exp = AB[:, np.newaxis, :, :]
    BC_exp = BC[:, np.newaxis, :, :]

    # Compute dot products using broadcasting and summation
    dot_AB_AM = np.sum(AB_exp * AM, axis=-1)
    dot_AB_AB = np.sum(AB * AB, axis=-1)
    dot_BC_BM = np.sum(BC_exp * BM, axis=-1)
    dot_BC_BC = np.sum(BC * BC, axis=-1)

    # Check using dot products
    mask = (
        (0 <= dot_AB_AM)
        & (dot_AB_AM <= dot_AB_AB[:, np.newaxis, :])
        & (0 <= dot_BC_BM)
        & (dot_BC_BM <= dot_BC_BC[:, np.newaxis, :])
    )

    return np.any(mask, axis=1)  # Collapse the corners axis


def check_rect_inside_rects(rects1, rects2):
    min_corners1 = rects1[:, 0, :]
    max_corners1 = rects1[:, 2, :]
    min_corners2 = rects2[:, 0, :]
    max_corners2 = rects2[:, 2, :]

    # Expand dimensions for broadcasting
    min_corners1_exp = min_corners1[:, np.newaxis, :]
    max_corners1_exp = max_corners1[:, np.newaxis, :]
    min_corners2_exp = min_corners2[np.newaxis, :, :]
    max_corners2_exp = max_corners2[np.newaxis, :, :]

    # Check containment conditions
    mask = (min_corners1_exp >= min_corners2_exp) & (
        max_corners1_exp <= max_corners2_exp
    )

    return mask[:, :, 0] & mask[:, :, 1]


def collides(region1_poses, region1_shapes, region2_poses, region2_shapes):
    corners1 = get_corners(region1_poses, region1_shapes)  # Nx4x2
    corners2 = get_corners(region2_poses, region2_shapes)  # Mx4x2

    assert len(corners1) > 0 and len(corners2) > 0

    collision_corners1 = check_corners_inside_rects(corners1, corners2)
    collision_corners2 = check_corners_inside_rects(
        corners2, corners1
    ).T  # Transpose to make it NxM
    collision_rect1 = check_rect_inside_rects(corners1, corners2)
    collision_rect2 = check_rect_inside_rects(
        corners2, corners1
    ).T  # Transpose to make it NxM

    return collision_corners1 | collision_corners2 | collision_rect1 | collision_rect2


def collides_regions(region1: Region, region2: Region) -> bool:
    region1_pose = np.array([[region1.pose.x, region1.pose.y, region1.pose.theta]])
    region2_pose = np.array([[region2.pose.x, region2.pose.y, region2.pose.theta]])

    region1_shape = np.array([[region1.shape.width, region1.shape.height]])
    region2_shape = np.array([[region2.shape.width, region2.shape.height]])

    return np.any(collides(region1_pose, region1_shape, region2_pose, region2_shape))


def apply_action_with_noise(
    state_array: np.array, action: SlamAction, noise_std_dev: float = ACTION_STD
) -> np.array:
    # The assumption here is that the robot's pose is the first three elements of each row of state_array
    robot_poses = state_array[:, :3]

    # Add Gaussian noise to the action's delta_x and delta_y
    noisy_delta_x = action.delta_x + np.random.normal(
        0, noise_std_dev, size=robot_poses.shape[0]
    )
    noisy_delta_y = action.delta_y + np.random.normal(
        0, noise_std_dev, size=robot_poses.shape[0]
    )

    # Calculate the new x and y positions based on noisy delta_x and delta_y
    new_xs = robot_poses[:, 0] + noisy_delta_x
    new_ys = robot_poses[:, 1] + noisy_delta_y

    # Calculate the new theta based on the direction of movement (delta_x, delta_y)
    new_thetas = np.arctan2(noisy_delta_y, noisy_delta_x)

    # Ensure theta is between -pi and pi
    new_thetas = np.mod(new_thetas + np.pi, 2 * np.pi) - np.pi

    # Assuming clamp is element-wise, apply clamp. If not, adjust accordingly
    clamped_xs = clamp(new_xs)
    clamped_ys = clamp(new_ys)

    # Update state_array with new poses
    new_state_array = copy.deepcopy(state_array)
    new_state_array[:, 0] = clamped_xs
    new_state_array[:, 1] = clamped_ys
    new_state_array[:, 2] = new_thetas

    # Update each of the target poses
    num_targets = (state_array.shape[1] - 3) // 6
    for t_index in range(num_targets):
        robot_T_target = new_state_array[
            :, (t_index + num_targets + 1) * 3 : (t_index + num_targets + 2) * 3
        ]
        if np.linalg.norm(robot_T_target) > 0.001:
            new_state_array[:, (t_index + 1) * 3 : (t_index + 2) * 3] = multiply_poses(
                new_state_array[:, :3],
                robot_T_target,
            )

    # TODO: Handle targets with attachments in a similar parallelized manner if possible.
    return new_state_array


def wrap_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def rrt_with_action_noise(
    particles: List[SlamState],  # Shape: (N, 3) for [x, y, theta]
    goal: Region,
    max_iterations: int = 2000,
    GOAL_SAMPLING_PROB: float = 0.2,
    std=0.0,
) -> Tuple[List[Action], AliasStore]:
    # Convert SlamState particles to numpy poses
    particles_poses = np.array([particle.to_array() for particle in particles])

    # Extract obstacle poses and shapes from the first particle (assuming static world)
    obstacles_poses, obstacles_shapes = get_vectorized_obstacles(particles[0])
    tree = {tuple(particles_poses[0]): None}
    tree_particles = {tuple(particles_poses[0]): particles_poses}
    new_goal = copy.deepcopy(goal)
    action_tree = {}

    robot_shape = ROBOT_SHAPE  # Assuming this global variable
    solved = False
    for it in range(max_iterations):
        if random.random() < GOAL_SAMPLING_PROB:
            rand_pose = np.array(
                [goal.sample().pose.x, goal.sample().pose.y, goal.sample().pose.theta]
            )
        else:
            rand_pose = np.array(
                [
                    np.random.random() * ENV_SIZE,
                    np.random.random() * ENV_SIZE,
                ]
            )

        nearest_pose_tuple, nearest_particles = min(
            ((tuple(pose), tree_particles[tuple(pose)]) for pose in tree.keys()),
            key=lambda x: distance_2d(np.array(x[0]), rand_pose),
        )
        nearest_pose = np.array(nearest_pose_tuple)

        # Calculate deltas
        delta_x = np.clip(rand_pose[0] - nearest_pose[0], -STEP_SIZE, STEP_SIZE)
        delta_y = np.clip(rand_pose[1] - nearest_pose[1], -STEP_SIZE, STEP_SIZE)

        # Calculate the new pose after applying the deltas
        new_x = nearest_pose[0] + delta_x
        new_y = nearest_pose[1] + delta_y
        new_theta = np.arctan2(
            delta_y, delta_x
        )  # Set theta based on direction of movement
        new_pose = np.array([new_x, new_y, new_theta])

        if tuple(new_pose) in action_tree or tuple(new_pose) in tree:
            continue

        # Define the action using delta_x and delta_y only
        candidate_action = SlamAction(delta_x, delta_y)

        new_particles_poses = apply_action_with_noise(
            nearest_particles, candidate_action, noise_std_dev=std
        )

        all_clear = True
        if obstacles_poses.shape[0] > 0:
            if np.any(
                collides(
                    new_particles_poses,
                    np.array(
                        [[robot_shape.width, robot_shape.height]]
                        * new_particles_poses.shape[0]
                    ),
                    obstacles_poses,
                    obstacles_shapes,
                )
            ):
                all_clear = False
        if all_clear:
            tree[tuple(new_pose)] = tuple(nearest_pose)
            tree_particles[tuple(new_pose)] = new_particles_poses
            action_tree[tuple(new_pose)] = candidate_action

            if distance_2d(tuple(new_pose), goal.pose.to_array()) < 0.01:
                solved = True
                new_goal.pose = Pose(new_pose[0], new_pose[1], new_pose[2])
                break

    current_pose_tuple = (new_goal.pose.x, new_goal.pose.y, new_goal.pose.theta)
    actions = []
    if solved:
        while current_pose_tuple is not None:
            parent_pose = tree.get(current_pose_tuple)
            if parent_pose is not None:
                action_taken = action_tree[current_pose_tuple]
                actions.append(action_taken)

            current_pose_tuple = parent_pose

        actions.reverse()

    return solved, actions


def angle_diff(angle1, angle2):
    # Normalizing angles to be between 0 and 2*pi radians
    angle1 = angle1 % (2 * math.pi)
    angle2 = angle2 % (2 * math.pi)

    # Compute the difference
    diff = abs(angle1 - angle2)

    # Find the smallest difference
    if diff > math.pi:
        diff = 2 * math.pi - diff

    return abs(diff)


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + angle_diff(p1[2], p2[2]) ** 2
    )


def distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def unsqueeze(arr: np.ndarray, axis=0):
    return np.expand_dims(arr, axis=axis)


def get_vectorized_obstacles(particle):
    obstacles_poses = np.array(
        [
            [obstacle.pose.x, obstacle.pose.y, obstacle.pose.theta]
            for obstacle in particle.obstacles
        ]
    )

    obstacles_shapes = np.array(
        [[obs.shape.width, obs.shape.height] for obs in particle.obstacles]
    )
    return obstacles_poses, obstacles_shapes


def generate_trajectory(particles, actions, std=ACTION_STD, vis=False):
    particle_array = np.array([p.to_array() for p in particles])
    particle_set_traj = [particle_array]
    obstacles_poses, obstacles_shapes = get_vectorized_obstacles(particles[0])

    for a in actions:
        particle_set_array = particle_set_traj[-1]
        new_particle_set_array = apply_action_with_noise(
            particle_set_array, a, noise_std_dev=std
        )
        particle_set_traj.append(new_particle_set_array)

    if not vis and len(actions) > 0:
        # We don't need the intermediate points if we aren't visualizing
        postprocess_actions = [actions[-1]]
        particle_set_traj = [particle_array, particle_set_traj[-1]]
    else:
        postprocess_actions = actions

    trajectories = [particles]
    for a, traj in zip(postprocess_actions, particle_set_traj[1:]):
        # TODO: determine if collision
        trajectory = []
        for start_state, new_state_vec in zip(particles, traj):
            new_state = start_state.from_array(new_state_vec)
            if a.attach:
                for target in new_state.targets:
                    if collides_regions(new_state.robot, target):
                        world_T_robot = unsqueeze(new_state.robot.pose.to_array())
                        world_T_target = unsqueeze(target.pose.to_array())
                        robot_T_target = Pose.from_array(
                            multiply_poses(
                                inverse(world_T_robot), world_T_target
                            ).squeeze()
                        )
                        new_state.attachments.append(
                            Attachment(target.name, robot_T_target)
                        )
            if a.detach:
                new_state.attachments = []

            if obstacles_poses.shape[0] > 0:
                if any(
                    collides(
                        unsqueeze(new_state.robot.pose.to_array()),
                        unsqueeze(ROBOT_SHAPE.to_array()),
                        obstacles_poses,
                        obstacles_shapes,
                    )[0]
                ):
                    new_state.in_collision = True

            trajectory.append(new_state)

        trajectories.append(trajectory)

    trajectories = [
        [particle_set[i] for particle_set in trajectories]
        for i in range(len(trajectories[0]))
    ]

    observations = []
    for trajectory in trajectories:
        ins = []
        for region in trajectory[-1].regions:
            if collides_regions(trajectory[-1].robot, region):
                ins.append(region.name)

        holding = None
        if len(trajectory[-1].attachments) > 0:
            holding = trajectory[-1].attachments[0]

        collided = any([s.in_collision for s in trajectory])
        observations.append(
            SlamObservation(
                regions_in=ins, holding=holding, primitives=actions, collision=collided
            )
        )
    return trajectories, observations


def get_robot_verts(robot_pose: Pose, robot_size: Size):
    # Get the four vertices of the rectangle
    corners = get_corners(
        unsqueeze(robot_pose.to_array()),
        unsqueeze(robot_size.to_array()),
    )[0]

    # Choose the top side's midpoint
    top_midpoint = (corners[0] + corners[3]) / 2

    # The two bottom corners
    bottom_left = corners[1]
    bottom_right = corners[2]

    # Construct the isosceles triangle
    triangle_verts = np.array([top_midpoint, bottom_left, bottom_right])

    return triangle_verts


def plot_trajectories(trajectories, save_dir, show_names=False):
    fig, ax = plt.subplots()
    frames_dir = os.path.join(save_dir, "frames")
    initial_num_files = len(
        [
            name
            for name in os.listdir(frames_dir)
            if os.path.isfile(os.path.join(frames_dir, name))
        ]
    )
    for i, states in enumerate(zip(*trajectories)):
        ax.clear()
        ax.set_xlim(0, ENV_SIZE)
        ax.set_ylim(0, ENV_SIZE)

        goal_shape = Circle(
            (states[0].goal.pose.x, states[0].goal.pose.y),
            radius=states[0].goal.shape.width / 2.0,
            fc="green",
            alpha=0.5,
        )
        ax.add_patch(goal_shape)

        # Draw 10x10 boundary box
        boundary = Polygon(
            [[0, 0], [ENV_SIZE, 0], [ENV_SIZE, ENV_SIZE], [0, ENV_SIZE]],
            closed=True,
            linewidth=1,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(boundary)

        for state in states:
            for idx, obstacle in enumerate(state.obstacles):
                vertices = get_corners(
                    unsqueeze(obstacle.pose.to_array()),
                    unsqueeze(obstacle.shape.to_array()),
                )[0]
                obstacle_poly = Polygon(
                    vertices,
                    closed=True,
                    linewidth=1,
                    facecolor="grey",
                    edgecolor="black",
                    alpha=0.5,
                )
                ax.add_patch(obstacle_poly)
                if show_names:
                    ax.text(
                        obstacle.pose.x,
                        obstacle.pose.y,
                        str(idx),
                        ha="center",
                        va="center",
                        color="white",
                    )

            for idx, target in enumerate(state.targets):
                vertices = get_corners(
                    unsqueeze(target.pose.to_array()),
                    unsqueeze(TARGET_SHAPE.to_array()),
                )[0]
                target_poly = Polygon(
                    vertices,
                    closed=True,
                    linewidth=1,
                    facecolor="yellow",
                    edgecolor="black",
                    alpha=0.5,
                )
                ax.add_patch(target_poly)
                if show_names:
                    ax.text(
                        target.pose.x,
                        target.pose.y,
                        str(idx),
                        ha="center",
                        va="center",
                        color="black",
                    )

            for idx, beacon in enumerate(state.beacons):
                circle = Circle(
                    (beacon.pose.x, beacon.pose.y),
                    radius=BEACON_RADIUS,
                    fc="blue",
                    edgecolor="black",
                )
                ax.add_patch(circle)
                if show_names:
                    ax.text(
                        beacon.pose.x,
                        beacon.pose.y,
                        str(idx),
                        ha="center",
                        va="center",
                        color="white",
                    )
                vision_circle = Circle(
                    (beacon.pose.x, beacon.pose.y),
                    radius=VISION_RADIUS,
                    fc="blue",
                    alpha=0.01,
                )
                ax.add_patch(vision_circle)

            robot_pose = state.robot.pose
            robot_size = state.robot.shape
            vertices = get_robot_verts(robot_pose, robot_size)
            arrow = Polygon(
                vertices,
                closed=True,
                linewidth=1,
                facecolor="purple",
                edgecolor="black",
                alpha=0.5,
            )
            ax.add_patch(arrow)
        frame_num = i + initial_num_files
        plt.gca().set_aspect("equal", adjustable="box")
        plot_fn = f"frame_{(frame_num):03d}.png"
        plt.savefig(os.path.join(frames_dir, plot_fn))


def generate_random_pose():
    return Pose(
        np.random.uniform(0, ENV_SIZE),
        np.random.uniform(0, ENV_SIZE),
        np.random.uniform(-math.pi, math.pi),
    )


def move_pick(
    action: Action, belief: SlamBelief, store: AliasStore
) -> AbstractBeliefSet:
    (target_index,) = store.get_all(action.args)
    particles = belief.particles

    mean_particle = copy.deepcopy(random.choice(particles))
    mean_particle.robot.pose = Pose.from_array(
        np.mean(np.array([p.robot.pose.to_array() for p in particles]), axis=0)
    )
    solved, actions = rrt_with_action_noise(
        [mean_particle], goal=particles[0].targets[target_index], std=0
    )
    if not solved:
        return []
    actions[-1].attach = True  # Attach
    return actions


def move_to(action: Action, belief: SlamBelief, store: AliasStore) -> AbstractBeliefSet:
    (region_index,) = store.get_all(action.args)
    particles = belief.particles
    mean_particle = copy.deepcopy(random.choice(particles))
    mean_particle.robot.pose = Pose.from_array(
        np.mean(np.array([p.robot.pose.to_array() for p in particles]), axis=0)
    )
    solved, actions = rrt_with_action_noise(
        [mean_particle], goal=particles[0].regions[region_index], std=0
    )

    if not solved:
        return None
    return actions


def move_look(
    action: Action, belief: SlamBelief, store: AliasStore
) -> AbstractBeliefSet:
    (region_index,) = store.get_all(action.args)
    particles = belief.particles
    mean_particle = copy.deepcopy(random.choice(particles))
    mean_particle.robot.pose = Pose.from_array(
        np.mean(np.array([p.robot.pose.to_array() for p in particles]), axis=0)
    )
    solved, actions = rrt_with_action_noise(
        [mean_particle], goal=particles[0].regions[region_index], std=0
    )
    if not solved:
        return None
    return actions


def move_place(
    action: Action, belief: SlamBelief, store: AliasStore
) -> AbstractBeliefSet:
    (target_index, region_index) = store.get_all(action.args)
    particles = belief.particles

    mean_particle = copy.deepcopy(random.choice(particles))
    mean_particle.robot.pose = Pose.from_array(
        np.mean(np.array([p.robot.pose.to_array() for p in particles]), axis=0)
    )
    solved, actions = rrt_with_action_noise(
        [mean_particle], goal=particles[0].regions[region_index], std=0
    )

    if not solved:
        return [SlamAction(0, 0, detach=True)]

    actions[-1].detach = True  # Detach
    return actions


def normalize_angle(angle):
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle


def move_corner(
    action: Action, belief: SlamBelief, store: AliasStore
) -> AbstractBeliefSet:
    (region_index,) = store.get_all(action.args)
    particles = copy.deepcopy(belief.particles)
    prev_means = np.array([np.inf, np.inf, np.inf])
    means = np.array([-np.inf, -np.inf, -np.inf])
    corner = particles[0].regions[region_index]
    actions = []
    num_steps = 0

    convergence_threshold = 0.00
    alignment_threshold = 0.1
    close_to_target_threshold = 1.0  # if robot is closer than this, reduce step size

    def distance_to_target(x, y, corner):
        return np.sqrt((corner.pose.x - x) ** 2 + (corner.pose.y - y) ** 2)

    actions = []
    particle_arrays = np.array([p.to_array() for p in particles])
    while np.linalg.norm(prev_means - means) > convergence_threshold:
        prev_means = means

        mean_y = np.mean(particle_arrays[:, 1])
        mean_x = np.mean(particle_arrays[:, 0])

        # Convert thetas to their respective x and y components

        mean_cos = np.mean(np.cos(particle_arrays[:, 2]))
        mean_sin = np.mean(np.sin(particle_arrays[:, 2]))
        mean_theta = np.arctan2(mean_sin, mean_cos)

        means = np.array([mean_x, mean_y, mean_theta])

        target_theta = atan2(corner.pose.y - mean_y, corner.pose.x - mean_x)
        delta_theta = normalize_angle(target_theta - mean_theta)

        distance_to_corner = distance_to_target(mean_x, mean_y, corner)

        # If the robot is not aligned to the corner, rotate it
        if abs(delta_theta) > alignment_threshold:
            new_delta_theta = max(min(delta_theta, MAX_ROTATION), -MAX_ROTATION)
            forward_step = 0  # Don't move forward, just rotate
        else:
            new_delta_theta = 0  # No need for rotation
            forward_step = (
                STEP_SIZE
                if distance_to_corner > close_to_target_threshold
                else distance_to_corner * 0.5
            )  # Reduce step size if very close

        action = SlamAction(forward_step, new_delta_theta)
        particle_arrays = apply_action_with_noise(particle_arrays, action)

        actions.append(action)
        num_steps += 1

        if num_steps > 100:
            break

    return actions


class SlamCollectEnv(TampuraEnv):
    def __init__(self, *args, num_particles=20, **kwargs):
        self.num_obstacles = 5
        self.num_beacons = 5
        self.num_targets = 3
        super(SlamCollectEnv, self).__init__(*args, **kwargs)
        self.state = None
        self.num_particles = num_particles

    def generate_initial_state(
        self, robot_pose: Pose, No: int, Nb: int, Nt: int, store: AliasStore
    ) -> SlamState:
        goal = Region(
            pose=Pose(8.0, 8.0, 0.0),
            shape=Size(GOAL_RADIUS * 2, GOAL_RADIUS * 2),
            exact=True,
        )
        start = Region(
            pose=copy.deepcopy(robot_pose),
            shape=Size(0.01, 0.01),
            exact=True,
        )
        obstacles = []
        beacons = []
        targets = []

        all_poses = [robot_pose, goal.pose]
        all_shapes = [ROBOT_SHAPE, goal.shape]

        # Generate obstacle poses
        for i in range(No):
            shape = Size(np.random.uniform(0.3, 0.6), np.random.uniform(0.3, 0.6))
            while True:
                pose = Pose(
                    np.random.uniform(0, ENV_SIZE),
                    np.random.uniform(0, ENV_SIZE),
                    np.random.uniform(-math.pi, math.pi),
                )
                obstacle = Region(pose, shape)
                if all(
                    not collides_regions(
                        obstacle, Region(existing_pose, existing_shape)
                    )
                    for existing_pose, existing_shape in zip(all_poses, all_shapes)
                ):
                    obstacles.append(obstacle)
                    all_poses.append(pose)
                    all_shapes.append(shape)
                    break

        # Generate beacon poses
        for _ in range(Nb):
            while True:
                pose = Pose(
                    np.random.uniform(0, ENV_SIZE),
                    np.random.uniform(0, ENV_SIZE),
                    np.random.uniform(-math.pi, math.pi),
                )
                beacon = Region(pose, VISION_SHAPE)
                if all(
                    not collides_regions(beacon, Region(existing_pose, existing_shape))
                    for existing_pose, existing_shape in zip(all_poses, all_shapes)
                ):
                    beacons.append(beacon)
                    all_poses.append(pose)
                    all_shapes.append(BEACON_SHAPE)
                    break

        # Generate target poses
        for _ in range(Nt):
            while True:
                pose = Pose(
                    np.random.uniform(0, ENV_SIZE),
                    np.random.uniform(0, ENV_SIZE),
                    np.random.uniform(-math.pi, math.pi),
                )
                target = Region(pose, TARGET_SHAPE)
                if all(
                    not collides_regions(target, Region(existing_pose, existing_shape))
                    for existing_pose, existing_shape in zip(all_poses, all_shapes)
                ):
                    targets.append(target)
                    all_poses.append(pose)
                    all_shapes.append(TARGET_SHAPE)
                    break

        corners = [
            Region(
                Pose(corner[0], corner[1], 0),
                Size(0.01, 0.01),
                name="corner_" + str(ci),
            )
            for ci, corner in enumerate(
                [
                    (0, 0),
                    (0, ENV_SIZE),
                    (ENV_SIZE, 0),
                    (ENV_SIZE, ENV_SIZE),
                ]
            )
        ]
        state = SlamState(
            Region(robot_pose, ROBOT_SHAPE),
            obstacles,
            beacons,
            targets,
            corners,
            [],
            goal,
            start,
        )
        return (state, store)

    def initialize(self) -> Tuple[Belief, AliasStore]:
        store = AliasStore()
        if os.path.exists(os.path.join(self.save_dir, "frames")):
            shutil.rmtree(os.path.join(self.save_dir, "frames"))
        os.makedirs(os.path.join(self.save_dir, "frames"))

        if self.state is None:
            self.starting_pose = Pose(1.0, 1.0, 1.0)

            self.state, store = self.generate_initial_state(
                self.starting_pose,
                self.num_obstacles,
                self.num_beacons,
                self.num_targets,
                store,
            )

            plot_trajectories([[self.state]], self.save_dir)

            self.starting_targets = self.state.targets

        self.state.robot_pose = self.starting_pose
        self.state.starting_targets = self.starting_targets

        for i, (region, subtype) in enumerate(
            zip(self.state.regions, self.state.subtypes)
        ):
            region.name = store.set(f"{OBJ}{subtype}_{i}", i, "region")

        for i, target in enumerate(self.state.targets):
            target.name = store.set(f"{OBJ}target_{i}", i, "target")

        self.objects = [r.name for r in self.state.regions + self.state.targets]
        self.object_types = ["region"] * len(self.state.regions) + ["target"] * len(
            self.state.targets
        )

        store.certified += [
            Atom(f"is_{subtype}", [self.state.regions[i].name])
            for i, subtype in enumerate(self.state.subtypes)
        ]

        obs = SlamObservation(initial_state=self.state)
        return SlamBelief(obs, self.num_particles, store), store

    def wrapup(self):
        logging.debug("Wrapping up slam collect env")
        frames_dir = Path(self.save_dir) / "frames"

        # ImageMagick: -delay 5  →  0.05 s per frame (5 × 1/100 s)
        delay_s = 0.05
        loop_forever = 0  # same as -loop 0

        png_paths = sorted(frames_dir.glob("*.png"))  # keep original order
        if not png_paths:
            logging.warning("No PNGs found in %s", frames_dir)
            return

        images = [iio.imread(p) for p in png_paths]
        out_gif = Path(self.save_dir) / "generated.gif"
        iio.imwrite(out_gif, images, duration=delay_s, loop=loop_forever)

        logging.debug("GIF written to %s (%d frames)", out_gif, len(images))

    def get_problem_spec(self) -> ProblemSpec:
        predicates = [
            Predicate("in", ["region"]),
            Predicate("target_in", ["target", "region"]),
            Predicate("holding", ["target"]),
            Predicate("is_beacon", ["region"]),
            Predicate("is_goal", ["region"]),
            Predicate("is_start", ["region"]),
            Predicate("is_corner", ["region"]),
            Predicate("in_collision", []),
            Predicate("known_pose", []),
        ]

        action_schemas = [
            ActionSchema(
                "move_pick",
                inputs=["?t"],
                input_types=["target"],
                depends=[Atom("known_pose")],
                preconditions=[
                    Not(Exists(Atom("holding", ["?t2"]), ["?t2"], ["target"])),
                ],
                effects=[ForAll(Not(Atom("in", ["?r2"])), ["?r2"], ["region"])],
                verify_effects=[Atom("holding", ["?t"]), Atom("known_pose")],
                effects_fn=effects_wrapper(move_pick),
                execute_fn=execute_wrapper(move_pick, self.save_dir, self.vis),
            ),
            ActionSchema(
                "move_place",
                inputs=["?t", "?r"],
                input_types=["target", "region"],
                depends=[Atom("known_pose")],
                preconditions=[
                    Atom("holding", ["?t"]),
                    Not(Atom("in_collision")),
                ],
                effects=[
                    ForAll(
                        When(Not(Eq("?r", "?r2")), Not(Atom("in", ["?r2"]))),
                        ["?r2"],
                        ["region"],
                    ),
                    Not(Atom("holding", ["?t"])),
                ],
                verify_effects=[
                    Atom("in", ["?r"]),
                    Atom("target_in", ["?t", "?r"]),
                    Atom("in_collision"),
                ],
                effects_fn=effects_wrapper(move_place),
                execute_fn=execute_wrapper(move_place, self.save_dir, self.vis),
            ),
            ActionSchema(
                "move_to",
                inputs=["?r"],
                input_types=["region"],
                depends=[Atom("known_pose")],
                preconditions=[
                    Not(Atom("is_beacon", ["?r"])),
                    Not(Atom("is_corner", ["?r"])),
                    Not(Atom("in_collision")),
                ],
                effects=[
                    ForAll(
                        When(Not(Eq("?r", "?r2")), Not(Atom("in", ["?r2"]))),
                        ["?r2"],
                        ["region"],
                    )
                ],
                verify_effects=[Atom("in", ["?r"]), Atom("in_collision")],
                effects_fn=effects_wrapper(move_to),
                execute_fn=execute_wrapper(move_to, self.save_dir, self.vis),
            ),
            ActionSchema(
                "move_look",
                inputs=["?r"],
                input_types=["region"],
                depends=[Atom("known_pose")],
                preconditions=[
                    Atom("is_beacon", ["?r"]),
                    Not(Atom("in_collision")),
                ],
                effects=[
                    ForAll(
                        When(Not(Eq("?r", "?r2")), Not(Atom("in", ["?r2"]))),
                        ["?r2"],
                        ["region"],
                    )
                ],
                verify_effects=[
                    Atom("in", ["?r"]),
                    Atom("known_pose"),
                    Atom("in_collision"),
                ],
                effects_fn=effects_wrapper(move_look),
                execute_fn=execute_wrapper(move_look, self.save_dir, self.vis),
            ),
            ActionSchema(
                "move_corner",
                inputs=["?c"],
                input_types=["region"],
                preconditions=[Atom("is_corner", ["?c"]), Not(Atom("in_collision"))],
                depends=[Atom("known_pose")],
                effects=[
                    ForAll(
                        When(Not(Eq("?c", "?r2")), Not(Atom("in", ["?r2"]))),
                        ["?r2"],
                        ["region"],
                    )
                ],
                verify_effects=[
                    Atom("known_pose"),
                    Atom("in", ["?c"]),
                    Atom("in_collision"),
                ],
                effects_fn=effects_wrapper(move_corner),
                execute_fn=execute_wrapper(move_corner, self.save_dir, self.vis),
            ),
            NoOp(),
        ]

        # Go straight to goal
        # reward = Exists(And([Atom("in", ["?r"]), Atom("is_goal", ["?r"])]), ["?r"], ["region"])

        # All objects in the goal
        reward = self.get_goal()

        return ProblemSpec(
            predicates=predicates,
            action_schemas=action_schemas,
            reward=reward,
        )

    def get_goal(self):
        return And(
            [
                ForAll(
                    Imply(Atom("is_goal", ["?r"]), Atom("target_in", ["?t", "?r"])),
                    ["?r", "?t"],
                    ["region", "target"],
                ),
                Not(Atom("in_collision")),
            ]
        )


class SlamCollectSimpleEnv(SlamCollectEnv):
    def __init__(self, *args, **kwargs):
        super(SlamCollectSimpleEnv, self).__init__(*args, **kwargs)
        self.num_obstacles = 20

    def get_goal(self):
        return And(
            [
                Exists(
                    And([Atom("is_goal", ["?r"]), Atom("in", ["?r"])]),
                    ["?r"],
                    ["region"],
                ),
                Not(Atom("in_collision")),
            ]
        )


class SlamCollectCustomEnv(SlamCollectEnv):
    def __init__(self, *args, **kwargs):
        super(SlamCollectCustomEnv, self).__init__(*args, **kwargs)
        self.num_obstacles = 20

    def generate_initial_state(
        self, robot_pose: Pose, No: int, Nb: int, Nt: int, store: AliasStore
    ) -> SlamState:
        goal = Region(
            pose=Pose(8.0, 8.0, 0.0),
            shape=Size(GOAL_RADIUS * 2, GOAL_RADIUS * 2),
            exact=True,
        )
        start = Region(
            pose=copy.deepcopy(robot_pose),
            shape=Size(0.01, 0.01),
            exact=True,
        )
        obstacles = []
        targets = []

        beacon_pose = Pose(ENV_SIZE / 2.0, ENV_SIZE / 2.0, 0)
        beacons = [Region(beacon_pose, VISION_SHAPE)]

        corners = []

        state = SlamState(
            Region(robot_pose, ROBOT_SHAPE),
            obstacles,
            beacons,
            targets,
            corners,
            [],
            goal,
            start,
        )
        return (state, store)

    def get_goal(self):
        return And(
            [
                Exists(
                    And([Atom("is_goal", ["?r"]), Atom("in", ["?r"])]),
                    ["?r"],
                    ["region"],
                ),
                Not(Atom("in_collision")),
            ]
        )


register_env("slam_collect", SlamCollectEnv)
register_env("slam_collect_simple", SlamCollectSimpleEnv)
register_env("slam_collect_custom", SlamCollectCustomEnv)

if __name__ == "__main__":
    import pickle

    pickle_file = "/Users/aidancurtis/tampura_environments/runs/run1724016024.424565/2024-08-18_17:29:53.pkl"
    file_path = os.path.join(pickle_file)
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    plot_trajectories([data.beliefs[-1].particles], save_dir=".", show_names=True)
