from __future__ import annotations

import copy
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
from tampura.config.config import register_env
from tampura.environment import TampuraEnv
from tampura.spec import ProblemSpec
from tampura.structs import (AbstractBelief, Action, ActionSchema, AliasStore,
                             Belief, NoOp, Observation, Predicate,
                             StreamSchema, effect_from_execute_fn)
from tampura.symbolic import And, Atom, Exists, Not, Or

import tampura_environments.panda_utils.pb_utils as pbu
from tampura_environments.panda_utils.panda_env_utils import (
    ARM_GROUP, CLIENT_MAP, GRIPPER_GROUP, OPEN_GRIPPER_POS, PANDA_TOOL_TIP,
    SELF_COLLISIONS, TABLE_AABB, TABLE_POSE, GroupTrajectory, SceneState,
    World, create_block, create_default_env, get_grasp, grasp_attachment,
    grasp_ik, pick_execute, place_execute, placement_sample, plan_motion,
    plan_workspace_motion, setup_robot_pybullet)
from tampura_environments.panda_utils.robot import (DEFAULT_ARM_POS, MISC_PATH,
                                                    PandaRobot)

NUM_PARTICLES = 10
KNOWN_POSE_STD = 0.01
NUM_SIM_STEPS = 5
BLOCK_SIZE = 0.02
INITIAL_STD = 0.02


class GhostSceneState(SceneState):
    def enforce_grasp(self):
        tool_link = pbu.link_from_name(
            self.world.robot, PANDA_TOOL_TIP, client=self.world.client
        )
        all_attach = [self.grasp.body]
        if self.grasp.body in self.world.ghosts:
            all_attach += self.world.ghosts[self.grasp.body]

        for ghost in all_attach:
            ghost_grasp = copy.deepcopy(self.grasp)
            ghost_grasp.body = ghost
            attachment = ghost_grasp.create_attachment(
                self.world.robot, link=tool_link, client=self.world.client
            )
            attachment.assign(client=self.world.client)


@dataclass
class HookWorld(World):
    ghosts: Dict[int, int] = field(default_factory=lambda: [])


@dataclass
class HookObservation(Observation):
    conf: List[float] = None
    grasp: str = None
    grasped_obj: str = None
    hook_traj: List[pbu.Pose] = field(default_factory=lambda: [])
    poses: Dict[str, pbu.Pose] = field(default_factory=lambda: {})


class HookBelief(Belief):
    def __init__(self, world, store, spec, init, pose_std=0.2):
        self.world = world
        self.conf = DEFAULT_ARM_POS
        self.grasp = None
        self.grasped_obj = None
        self.hook = self.world.objects[-1]

        # Create particle-based belief representations for each object pose
        self.ghost_dict = {
            o: self.world.ghosts[store.get(o)] for o in self.world.objects[:-1]
        }
        self.object_dists = {
            o: [
                pbu.get_pose(go, client=self.world.client)
                for go in self.world.ghosts[store.get(o)]
            ]
            for o in self.world.objects[:-1]
        }

        # We know the hook pose
        self.object_dists[self.hook] = [
            pbu.get_pose(store.get(self.hook), client=self.world.client)
        ]

    def vectorize(self):
        vecs = []
        for obj in self.world.objects:
            vecs += [self.get_com(obj), self.get_std(obj)]
        return np.concatenate(vecs)

    def simulate_hook_traj(self, hook_traj, store):
        # Set the obj and ghost poses
        for o, go_poses in self.object_dists.items():
            if o != self.hook:
                for go_i, go_pose in enumerate(go_poses):
                    pbu.set_pose(
                        self.ghost_dict[o][go_i], go_pose, client=self.world.client
                    )

        # Run the hook thorugh the sequence of poses
        for hook_pose in hook_traj:
            # wait_if_gui(client=self.world.client)
            pbu.set_pose(store.get(self.hook), hook_pose, client=self.world.client)
            _ = [self.world.client.stepSimulation() for _ in range(NUM_SIM_STEPS)]

        # Get the new object particle poses
        for o, gos in self.ghost_dict.items():
            self.object_dists[o] = [
                pbu.get_pose(go, client=self.world.client) for go in gos
            ]

    def get_com(self, obj):
        return np.mean(
            np.concatenate([np.array([pose[0]]) for pose in self.object_dists[obj]]),
            axis=0,
        )

    def get_std(self, obj):
        values = np.concatenate(
            [np.array([pose[0]]) for pose in self.object_dists[obj]]
        )
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        robust_std = 1.4826 * mad  # Scaling factor for normal distribution
        return robust_std

    def update(
        self, action: Action, obs: HookObservation, store: AliasStore
    ) -> HookBelief:
        new_belief = copy.deepcopy(self)

        if obs is None:
            return new_belief

        new_belief.conf = obs.conf
        new_belief.grasp = obs.grasp
        new_belief.grasped_obj = obs.grasped_obj

        for obj, known_pose in obs.poses.items():
            new_belief.object_dists[obj] = [known_pose]

        if len(obs.hook_traj) > 0:
            new_belief.simulate_hook_traj(obs.hook_traj, store)

        return new_belief

    def abstract(self, store: AliasStore) -> AbstractBelief:
        items = []
        if self.grasp is not None:
            items.append(Atom("at-grasp", [self.grasped_obj, self.grasp]))

        for o in self.object_dists:
            if all(self.get_std(o) < KNOWN_POSE_STD):
                items.append(Atom("known-pose", [o]))

        # Calculate on probability from object poses
        coms = {obj: self.get_com(obj) for obj in self.object_dists.keys()}
        for obj1, com1 in coms.items():
            for obj2, com2 in coms.items():
                xydist = np.linalg.norm(np.array(com1)[:2] - np.array(com2)[:2])
                if com1[2] > (com2[2] + BLOCK_SIZE / 4.0) and xydist < BLOCK_SIZE:
                    items.append(Atom("on", [obj1, obj2]))

        return AbstractBelief(items)

    def get_pose(self, obj):
        """Sample a pose from the belief set."""
        return random.choice(self.object_dists[obj])


def add_gaussian_distributed_cubes(
    mean_pose, std_dev, num_cubes=100, cube_size=BLOCK_SIZE, alpha=0.2, client=None
):
    """Add small cubes distributed according to a Gaussian in the x and y
    directions.

    :param mean_pose: Tuple (x, y, z) indicating the mean pose.
    :param std_dev: Standard deviation for Gaussian distribution.
    :param num_cubes: Number of cubes to generate.
    :param cube_size: Half-extent size of each cube (cubes are
        2*cube_size in each dimension).
    :return: List of cube IDs.
    """
    cubes = []

    # Red color with low alpha (transparency)
    rgbaColor = [1, 0.5, 0, alpha]

    for _ in range(num_cubes):
        x_offset = np.random.normal(0, std_dev)
        y_offset = np.random.normal(0, std_dev)
        z_offset = 0  # Assuming no variation in z-direction

        cube_position = (
            mean_pose[0] + x_offset,
            mean_pose[1] + y_offset,
            mean_pose[2] + z_offset,
        )

        cube_id = create_block(
            rgbaColor,
            cube_position,
            halfExtents=(cube_size, cube_size, cube_size),
            client=client,
        )
        cubes.append(cube_id)

    return cubes


def create_hook(client=None):
    hook_path = os.path.join(MISC_PATH, "hook/hook.urdf")
    hook = client.loadURDF(hook_path)
    return hook


def setup_world(means, vis=False) -> World:
    new_upper = list(TABLE_AABB.upper)
    EXTRA_LEN = 0.5
    new_upper[0] += EXTRA_LEN
    new_aabb = pbu.AABB(lower=TABLE_AABB.lower, upper=new_upper)
    new_pose = (
        pbu.Point(
            TABLE_POSE[0][0] - EXTRA_LEN / 2.0, TABLE_POSE[0][1], TABLE_POSE[0][2]
        ),
        TABLE_POSE[1],
    )

    robot_body, client = setup_robot_pybullet(gui=vis)

    robot = PandaRobot(robot_body, client=client)

    client_id = len(CLIENT_MAP)
    CLIENT_MAP[client_id] = client

    # Define unique collision groups
    GROUP_MAIN_CUBE = 1
    GROUP_GHOST = 2
    GROUP_FLOOR_HOOK = 4
    GROUP_OTHERS = 8

    # Define collision masks
    MASK_GHOST = GROUP_FLOOR_HOOK
    MASK_FLOOR_HOOK = GROUP_MAIN_CUBE | GROUP_GHOST | GROUP_OTHERS

    movable = []
    ghosts = {}

    floor, obstacles = create_default_env(
        client=client, table_aabb=new_aabb, table_pose=new_pose
    )
    client.setCollisionFilterGroupMask(
        floor, -1, GROUP_FLOOR_HOOK, MASK_FLOOR_HOOK
    )  # Set collision filter for floor

    hook = create_hook(client=client)
    new_hook_pose = pbu.Pose(
        point=pbu.Point(x=0.1, y=-0.35, z=0.02), euler=pbu.Euler(yaw=-math.pi / 6)
    )
    pbu.set_pose(hook, new_hook_pose, client=client)
    client.setCollisionFilterGroupMask(
        hook, -1, GROUP_FLOOR_HOOK, MASK_FLOOR_HOOK
    )  # Set collision filter for hook

    for mean in means:
        ghost_cubes = add_gaussian_distributed_cubes(
            mean_pose=mean,
            std_dev=INITIAL_STD,
            cube_size=BLOCK_SIZE,
            num_cubes=NUM_PARTICLES,
            alpha=2 / float(NUM_PARTICLES),
            client=client,
        )
        # Set collision filter for each ghost cube
        for ghost_cube in ghost_cubes:
            client.setCollisionFilterGroupMask(ghost_cube, -1, GROUP_GHOST, MASK_GHOST)

        ghosts[ghost_cubes[0]] = ghost_cubes
        movable += [ghost_cubes[0]]

    movable += [hook]

    pbu.set_joint_positions(
        robot_body,
        robot.get_group_joints(ARM_GROUP, client=client),
        DEFAULT_ARM_POS,
        client=client,
    )

    pbu.set_joint_positions(
        robot_body,
        robot.get_group_joints(GRIPPER_GROUP, client=client),
        OPEN_GRIPPER_POS,
        client=client,
    )

    client.setGravity(0, 0, -10)

    return HookWorld(
        client_id=client_id,
        robot=robot,
        environment=obstacles + movable,
        floor=floor,
        objects=movable,
        ghosts=ghosts,
    )


def get_hook_traj(origin: pbu.Pose):
    pre_hook_pose = pbu.multiply(pbu.Pose(pbu.Point(x=-0.05, y=-0.05, z=0.05)), origin)
    post_hook_pose_1 = pbu.multiply(pbu.Pose(pbu.Point(z=-0.05)), pre_hook_pose)
    post_hook_pose_2 = pbu.multiply(
        pbu.Pose(pbu.Point(x=-0.15, y=0.1)), post_hook_pose_1
    )
    post_hook_pose_3 = pbu.multiply(pbu.Pose(pbu.Point(z=0.05)), post_hook_pose_2)

    intermediate_hook_poses = (
        [pre_hook_pose]
        + list(
            pbu.interpolate_poses(pre_hook_pose, post_hook_pose_1, pos_step_size=0.01)
        )
        + list(
            pbu.interpolate_poses(
                post_hook_pose_1, post_hook_pose_2, pos_step_size=0.005
            )
        )
        + list(
            pbu.interpolate_poses(
                post_hook_pose_2, post_hook_pose_3, pos_step_size=0.01
            )
        )
    )
    return intermediate_hook_poses


def pull_execute_fn(
    action: Action, belief: HookBelief, state: GhostSceneState, store: AliasStore
) -> Tuple[GhostSceneState, Observation]:
    o1, g, o2 = action.args

    default_obs = HookObservation(conf=belief.conf, grasp=g, grasped_obj=o1)

    # Compute the hook pose based on the target object COM
    world_T_obj = pbu.Pose(pbu.Point(*belief.get_com(o2)))
    hook_traj = get_hook_traj(world_T_obj)
    grasp_poses = [
        pbu.multiply(pose, pbu.invert(store.get(g).grasp)) for pose in hook_traj
    ]
    robot = belief.world.robot
    pbu.set_joint_positions(
        robot,
        robot.get_group_joints(ARM_GROUP, client=belief.world.client),
        belief.conf,
        client=belief.world.client,
    )
    arm_path = plan_workspace_motion(
        robot,
        grasp_poses,
        obstacles=[belief.world.floor],
        client=belief.world.client,
    )
    arm_joints = pbu.joints_from_names(
        robot, robot.joint_groups[ARM_GROUP], client=belief.world.client
    )

    collision_fn = pbu.get_collision_fn(
        robot,
        arm_joints,
        obstacles=[belief.world.floor] + list(chain(*belief.world.ghosts.values())),
        attachments=[],
        self_collisions=SELF_COLLISIONS,
        disable_collisions=True,
        client=belief.world.client,
    )

    if arm_path is None:
        logging.debug("Pull ik fail")
        # wait_if_gui(client = belief.world.client)
        return state, default_obs

    if collision_fn(arm_path[0]):
        logging.debug("Pull ik collision")
        # wait_if_gui(client = belief.world.client)
        return state, default_obs

    if state is not None:
        motion_plan1 = plan_motion(
            belief.world,
            belief.conf,
            arm_path[0],
            obstacles=[belief.world.floor] + list(chain(*belief.world.ghosts.values())),
            attachments=[grasp_attachment(belief.world, store.get(g))],
        )

        if motion_plan1 is None:
            logging.debug("Pull motion plan fail")
            return state, default_obs

        _ = state.apply_sequence(
            [motion_plan1], teleport=not pbu.has_gui(client=state.world.client)
        )
        gt = GroupTrajectory(
            state.world.robot, ARM_GROUP, arm_path, client=state.world.client
        )
        _ = state.apply_sequence([gt], teleport=False, sim_steps=NUM_SIM_STEPS)

    obs = HookObservation(
        conf=arm_path[-1], grasp=g, grasped_obj=o1, hook_traj=hook_traj
    )

    return state, obs


def stack_execute_fn(
    action: Action, belief: HookBelief, state: GhostSceneState, store: AliasStore
) -> Tuple[GhostSceneState, Observation]:
    o1, g, o2 = action.args
    default_obs = HookObservation(conf=belief.conf, grasp=g, grasped_obj=o1)
    o2_pose = belief.get_pose(o2)
    stacked_pose = pbu.multiply(pbu.Pose(pbu.Point(z=2 * BLOCK_SIZE)), o2_pose)
    pre_confs = grasp_ik(
        belief.world,
        store.get(o1),
        stacked_pose,
        store.get(g),
        obstacles=[belief.world.floor],
    )
    if pre_confs is None:
        logging.debug("Stack ik fail")
        return state, default_obs

    if state is not None:
        motion_plan = plan_motion(
            belief.world,
            belief.conf,
            pre_confs[0],
            obstacles=[belief.world.floor],
            attachments=[grasp_attachment(belief.world, store.get(g))],
        )
        if motion_plan is None:
            logging.debug("Stack motion plan fail")
            return state, default_obs

        _ = place_execute(state, motion_plan, pre_confs)

    return state, HookObservation(
        conf=pre_confs[0], grasp=None, poses={o1: stacked_pose}
    )


def grasp_sample_fn_wrapper(world):
    def grasp_sample_fn(args: List[str], store: AliasStore):
        (obj,) = store.get_all(args)
        grasp = get_grasp(
            world,
            obj,
            world.environment,
            grasp_mode="mesh",
            use_saved=False,
            client=world.client,
        )
        return grasp

    return grasp_sample_fn


def placement_sample_fn_wrapper(world):
    def placement_sample_fn(args: List[str], store: AliasStore):
        (obj, region) = store.get_all(args)
        return placement_sample(world, obj, region)

    return placement_sample_fn


def pick_execute_fn(
    action: Action,
    belief: HookBelief,
    state: GhostSceneState,
    store: AliasStore,
) -> Tuple[GhostSceneState, Observation, AliasStore]:
    default_obs = HookObservation(conf=belief.conf)
    obj, g = action.args
    pre_confs = grasp_ik(
        belief.world,
        obj,
        belief.get_pose(obj),
        store.get(g),
        obstacles=[belief.world.floor],
    )
    if pre_confs is None:
        logging.debug("Pick ik fail")
        # wait_if_gui(client=belief.world.client)
        return state, default_obs

    if state is not None:
        motion_plan = plan_motion(
            belief.world,
            belief.conf,
            pre_confs[0],
            obstacles=[
                belief.world.floor
            ],  # + list(chain(*belief.world.ghosts.values())),
        )

        if motion_plan is None:
            logging.debug("Pick motion plan fail")
            return state, default_obs

        if state is not None:
            _ = pick_execute(state, store.get(g), motion_plan, pre_confs)

    observation = HookObservation(conf=pre_confs[0], grasp=g, grasped_obj=obj)
    return state, observation


def place_execute_fn(
    action: Action, belief: HookBelief, state: GhostSceneState, store: AliasStore
) -> Tuple[GhostSceneState, Observation]:
    o, p, g, _ = action.args
    default_obs = HookObservation(conf=belief.conf, grasp=g, grasped_obj=o)
    pre_confs = grasp_ik(
        belief.world,
        store.get(o),
        store.get(p),
        store.get(g),
        obstacles=[belief.world.floor],
    )
    if pre_confs is None:
        logging.debug("Place ik fail")
        return state, default_obs

    if state is not None:
        non_grasping_ghosts = list(
            chain(
                *[belief.ghost_dict[ghost] for ghost in belief.ghost_dict if ghost != o]
            )
        )
        motion_plan = plan_motion(
            belief.world,
            belief.conf,
            pre_confs[0],
            obstacles=[belief.world.floor],  # + non_grasping_ghosts,
            attachments=[grasp_attachment(belief.world, store.get(g))],
        )
        if motion_plan is None:
            logging.debug("Place motion plan fail")
            return state, default_obs

        _ = place_execute(state, motion_plan, pre_confs)

    return state, HookObservation(
        conf=pre_confs[0], grasp=None, poses={o: store.get(p)}
    )


def min_pairwise_distance(means):
    mean_distances = []
    for mean1 in means:
        for mean2 in means:
            if mean1 != mean2:
                mean_distances.append(np.linalg.norm(np.array(mean1) - np.array(mean2)))
    return min(mean_distances)


class ToolUseEnv(TampuraEnv):
    def initialize(self) -> Tuple[Belief, AliasStore]:
        store = AliasStore()

        while True:
            means = [
                [random.uniform(0.4, 0.7), random.uniform(-0.3, 0.3), 0.02]
                for _ in range(3)
            ]
            if min_pairwise_distance(means) > 10 * INITIAL_STD:
                break

        self.world = setup_world(means, vis=self.vis)
        self.sim_world = setup_world(means, vis=False)

        self.poses = [
            store.add_typed(pbu.get_pose(obj, client=self.sim_world.client), "pose")
            for obj in self.sim_world.objects
        ]
        self.sim_world.objects = [
            store.add_typed(cubes, "physical") for cubes in self.sim_world.objects
        ]

        self.region = store.add_typed(self.sim_world.floor, "region")
        init = []
        for obj, pose in zip(self.sim_world.objects, self.poses):
            store.certified.append(Atom("object-pose", [obj, pose]))

        init.append(Atom("known-pose", [self.sim_world.objects[-1]]))
        store.certified.append(Atom("hook", [self.sim_world.objects[-1]]))

        self.state = GhostSceneState(self.world)
        return HookBelief(self.sim_world, store, self.problem_spec, init), store

    def get_problem_spec(self) -> ProblemSpec:
        predicates = [
            Predicate("on", ["physical", "physical"]),
            Predicate("object-grasp", ["physical", "grasp"]),
            Predicate("at-grasp", ["physical", "grasp"]),
            Predicate("known-pose", ["physical"]),
            Predicate("placement", ["physical", "pose", "region"]),
            Predicate("hook", ["physical"]),
        ]

        stream_schemas = [
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
                name="placement-sample",
                inputs=["?o", "?r"],
                input_types=["physical", "region"],
                output="?p",
                output_type="pose",
                certified=[
                    Atom("placement", ["?o", "?p", "?r"]),
                ],
                sample_fn=placement_sample_fn_wrapper(self.sim_world),
            ),
        ]

        action_schemas = [
            ActionSchema(
                name="pick",
                inputs=["?o", "?g"],
                input_types=["physical", "grasp"],
                preconditions=[
                    Not(Exists(Atom("on", ["?o2", "?o"]), ["?o2"], ["physical"])),
                    Not(Exists(Atom("on", ["?o", "?o2"]), ["?o2"], ["physical"])),
                    Not(
                        Exists(
                            Atom("at-grasp", ["?o2", "?g2"]),
                            ["?o2", "?g2"],
                            ["physical", "grasp"],
                        )
                    ),
                    Atom("known-pose", ["?o"]),
                    Atom("object-grasp", ["?o", "?g"]),
                ],
                effects=[Atom("at-grasp", ["?o", "?g"])],
                execute_fn=pick_execute_fn,
                effects_fn=effect_from_execute_fn(pick_execute_fn),
            ),
            ActionSchema(
                name="place",
                inputs=["?o", "?p", "?g", "?r"],
                input_types=["physical", "pose", "grasp", "region"],
                preconditions=[
                    Atom("at-grasp", ["?o", "?g"]),
                    Atom("placement", ["?o", "?p", "?r"]),
                ],
                effects=[
                    Not(Atom("at-grasp", ["?o", "?g"])),
                    Atom("known-pose", ["?o"]),
                ],
                execute_fn=place_execute_fn,
                effects_fn=effect_from_execute_fn(place_execute_fn),
            ),
            ActionSchema(
                name="stack",
                inputs=["?o1", "?g1", "?o2"],
                input_types=["physical", "grasp", "physical"],
                preconditions=[
                    Not(Atom("hook", ["?o1"])),
                    Not(Atom("hook", ["?o2"])),
                    Not(Exists(Atom("on", ["?o3", "?o2"]), ["?o3"], ["physical"])),
                    Atom("at-grasp", ["?o1", "?g1"]),
                    Or(
                        [
                            Atom("known-pose", ["?o2"]),
                            Exists(Atom("on", ["?o2", "?o3"]), ["?o3"], ["physical"]),
                        ]
                    ),
                ],
                effects=[
                    Atom("on", ["?o1", "?o2"]),
                    Not(Atom("at-grasp", ["?o1", "?g1"])),
                ],
                execute_fn=stack_execute_fn,
                effects_fn=effect_from_execute_fn(stack_execute_fn),
            ),
            ActionSchema(
                name="pull-towards",
                inputs=["?o1", "?g1", "?o2"],
                input_types=["physical", "grasp", "physical"],
                preconditions=[
                    Atom("hook", ["?o1"]),
                    Atom("at-grasp", ["?o1", "?g1"]),
                    Not(Atom("known-pose", ["?o2"])),
                ],
                verify_effects=[Atom("known-pose", ["?o2"])],
                execute_fn=pull_execute_fn,
                effects_fn=effect_from_execute_fn(pull_execute_fn),
            ),
            NoOp(),
        ]

        reward = And(
            [
                Atom("on", [self.sim_world.objects[i], self.sim_world.objects[i + 1]])
                for i in range(len(self.sim_world.objects) - 2)
            ]
        )

        spec = ProblemSpec(
            predicates=predicates,
            stream_schemas=stream_schemas,
            action_schemas=action_schemas,
            reward=reward,
        )

        return spec


register_env("tool_use", ToolUseEnv)
