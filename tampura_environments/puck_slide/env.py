from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
from scipy.stats import multivariate_normal
from tampura.config.config import register_env
from tampura.environment import TampuraEnv
from tampura.spec import ProblemSpec
from tampura.structs import (AbstractBelief, AbstractBeliefSet, Action,
                             ActionSchema, AliasStore, Belief, NoOp,
                             Observation, StreamSchema)
from tampura.symbolic import OBJ, Atom, Not, Predicate

import tampura_environments.panda_utils.pb_utils as pbu
from tampura_environments.panda_utils.panda_env_utils import (
    ARM_GROUP, CLIENT_MAP, GRIPPER_GROUP, OPEN_GRIPPER_POS, PANDA_PATH,
    PandaRobot, SceneState, World, field, plan_motion, plan_workspace_motion,
    pose_to_vec)
from tampura_environments.panda_utils.primitives import follow_path
from tampura_environments.panda_utils.robot import (CLOSE_GRIPPER_POS,
                                                    DEFAULT_ARM_POS)

PUCK_RADIUS = 0.06
PUCK_HEIGHT = 0.04
PUCK_AABB = pbu.AABB(
    lower=[-PUCK_RADIUS, -PUCK_RADIUS, -PUCK_HEIGHT / 2.0],
    upper=[PUCK_RADIUS, PUCK_RADIUS, PUCK_HEIGHT / 2.0],
)
FLOOR_WIDTH = 1.1
FLOOR_OFFSET = FLOOR_WIDTH / 2.0 + 0.3
IK_RADIUS = 0.3
FLOOR_LENGTH = 0.7
FLOOR_POS = [FLOOR_OFFSET, 0, 0]
PUCK_START_POSE = pbu.Pose(
    pbu.Point(
        x=IK_RADIUS / 2.0 + FLOOR_OFFSET - FLOOR_WIDTH / 2.0 - PUCK_RADIUS, z=0.06
    )
)
GOAL_SEPARATION = 0.2
GOAL_POSITIONS = [
    [FLOOR_OFFSET + FLOOR_WIDTH / 2.0 - 0.1 - GOAL_SEPARATION * i, 0, 0.05]
    for i in range(3)
]
GOAL_AABB = pbu.AABB(
    lower=[-0.1, -FLOOR_LENGTH / 2.0, -0.1], upper=[0.1, FLOOR_LENGTH / 2.0, 0.1]
)

IK_POS = [FLOOR_OFFSET - FLOOR_WIDTH / 2.0 + IK_RADIUS, 0, 0.0]
IK_AABB = pbu.AABB(
    lower=[-IK_RADIUS, -FLOOR_LENGTH / 2.0, -0.1],
    upper=[IK_RADIUS, FLOOR_LENGTH / 2.0, 0.1],
)
FRICTION_LOWER = 0.05
FRICTION_UPPER = 0.15
FRICTION_MID = (FRICTION_LOWER + FRICTION_UPPER) / 2.0
PUCK_PARTICLES = 20

STOOL_POS = [0, 0, 0]
STOOL_AABB = pbu.AABB(
    lower=[-0.3, -FLOOR_LENGTH / 2.0, -0.1], upper=[0.3, FLOOR_LENGTH / 2.0, 0.1]
)
START_AABB = pbu.AABB(lower=[-0.065, -0.065, -0.1], upper=[0.065, 0.065, 0.1])
MASS = 1
GOAL_OBJECTS = [f"{OBJ}goal{i}" for i in range(1)]
REGIONS = [f"{OBJ}ik"] + GOAL_OBJECTS
OBJECTS = [f"{OBJ}puck"]
SHAPES = {f"{OBJ}puck": PUCK_AABB, f"{OBJ}ik": IK_AABB} | {
    k: GOAL_AABB for k in GOAL_OBJECTS
}
REGION_POSES = {
    obj: pbu.Pose(pos) for obj, pos in zip(GOAL_OBJECTS, GOAL_POSITIONS)
} | {f"{OBJ}ik": pbu.Pose(IK_POS)}


@dataclass
class PuckState(SceneState):
    world: World = None


def get_client_id():
    return len(CLIENT_MAP) + 1


def update_sim(objects, pose, frictions, client):
    for puck, particle_friction in zip(objects, frictions):
        pbu.set_pose(puck, pose, client=client)
        client.changeDynamics(puck, -1, lateralFriction=particle_friction)


@dataclass
class PuckObservation(Observation):
    puck_pose: pbu.Pose = None
    joint_path: List[List[float]] = field(default_factory=lambda: {})
    effort: float = 0
    unreachable: bool = False


class PuckSlideBelief(Belief):
    def __init__(
        self, world: World, obs: PuckObservation, store: AliasStore, *args, **kwargs
    ):
        self.puck_pose = obs.puck_pose
        self.world = world
        self.particle_frictions = np.array(
            [
                pbu.get_dynamics_info(puck, client=world.client).lateral_friction
                for puck in world.objects
            ]
        )

        self.particle_weights = np.array(
            [1.0 / len(world.objects) for _ in range(len(world.objects))]
        )

    def vectorize(self):
        return np.concatenate([pose_to_vec(self.puck_pose), self.particle_frictions])

    def __str__(self):
        return "Estimated friction: {} with std {}".format(
            self.estimated_friction(), self.get_friction_std()
        )

    def get_friction_std(self):
        frictions = np.array(self.particle_frictions)
        weights = np.array(self.particle_weights)
        weighted_mean = np.average(frictions, weights=weights)
        return np.sqrt(np.average((frictions - weighted_mean) ** 2, weights=weights))

    def abstract(self, store: AliasStore) -> AbstractBelief:
        """This one is deterministic in belief space, but not all of them need
        to be."""
        known = []

        if self.get_friction_std() < 0.01:
            known.append(Atom("known-friction", [f"{OBJ}puck"]))

        for region in REGIONS:
            if test_in(self.puck_pose, region):
                known.append(Atom("in", [f"{OBJ}puck", region]))

        if self.puck_pose[0][2] < 0:
            known.append(Atom("off-table"))

        return AbstractBelief(known)

    def update_particle_frictions(self, particle_poses, observed_pose):
        def pose_xy(pose: pbu.Pose) -> np.array:
            return np.array([pose[0][0], pose[0][1]])

        pose_likelihoods = [
            multivariate_normal.pdf(
                pose_xy(particle_pose),
                mean=pose_xy(observed_pose),
                cov=np.eye(2) * 1e-3,
            )
            for particle_pose in particle_poses
        ]

        if sum(pose_likelihoods) == 0:
            pose_likelihoods = [
                1.0 / len(particle_poses) for _ in range(len(particle_poses))
            ]

        new_weights = [
            li * pw for li, pw in zip(pose_likelihoods, self.particle_weights)
        ]
        posterior = [w / sum(new_weights) for w in new_weights]

        self.particle_weights = posterior

    def reweight_particles(self):
        N = len(self.particle_weights)  # Number of particles
        cumulative_sum = np.cumsum(self.particle_weights)
        cumulative_sum[-1] = 1.0  # Ensure the sum is exactly one
        positions = (np.arange(N) + np.random.random()) / N  # Deterministic intervals

        indexes = np.searchsorted(cumulative_sum, positions).astype(int)

        self.particle_weights = np.array(self.particle_weights)[indexes]
        self.particle_frictions = np.array(self.particle_frictions)[indexes]

        # Adding noise to particle frictions for diversity
        self.particle_frictions += np.random.normal(0, 0.0005, size=N)

        # Normalize the weights after resampling
        self.particle_weights /= np.sum(self.particle_weights)

    def update_sim(self):
        update_sim(
            self.world.objects,
            self.puck_pose,
            self.particle_frictions,
            self.world.client,
        )

    def estimated_friction(self):
        return np.average(self.particle_frictions, weights=self.particle_weights)

    def update(
        self,
        action: Action,
        observation: PuckObservation,
        store: AliasStore,
    ) -> PuckSlideBelief:
        new_belief = copy.deepcopy(self)

        if observation is None:
            return new_belief

        self.update_sim()
        new_particle_poses = follow_joint_path(
            self.world, joint_path=observation.joint_path, lead_step=observation.effort
        )

        new_belief.update_particle_frictions(new_particle_poses, observation.puck_pose)
        new_belief.reweight_particles()

        new_belief.puck_pose = observation.puck_pose
        return new_belief


def create_pillar(width=0.25, length=0.25, height=1e-3, color=None, **kwargs):
    return pbu.create_box(w=width, l=length, h=height, color=color, **kwargs)


def create_puck(friction, client, alpha=1.0):
    puck = pbu.create_cylinder(
        radius=PUCK_RADIUS,
        height=PUCK_HEIGHT,
        color=pbu.RGBA(0.2, 0.2, 0.2, alpha),
        mass=MASS,
        client=client,
    )
    client.changeDynamics(puck, -1, lateralFriction=friction)
    return puck


def setup_puck_world(pose, friction, vis=False) -> World:
    client_id = get_client_id()
    CLIENT_MAP[client_id] = bc.BulletClient(connection_mode=p.GUI if vis else p.DIRECT)
    client = CLIENT_MAP[client_id]
    client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    robot_body = pbu.load_pybullet(PANDA_PATH, fixed_base=True, client=client)

    robot = PandaRobot(robot_body, client=client)

    client.setGravity(0, 0, -9.8)

    floor = create_pillar(
        width=FLOOR_WIDTH,
        length=FLOOR_LENGTH,
        height=0.1,
        color=pbu.TAN,
        client=client,
    )

    for line_y in range(3):
        for number_index in range(3 - line_y):
            number = create_pillar(
                width=0.04,
                length=0.01,
                height=0.002,
                color=pbu.RGBA(0.0, 0.0, 0, 1.0),
                collision=True,
                client=client,
            )
            pos = copy.deepcopy(GOAL_POSITIONS[line_y])
            pos[0] -= 0.0
            pos[1] = pos[1] + number_index * 0.03 - 0.03 * (3 - line_y) / 2.0 + 0.01
            pbu.set_pose(number, pbu.Pose(pbu.Point(*pos)), client=client)

        line = create_pillar(
            width=0.01,
            length=GOAL_AABB.upper[1] - GOAL_AABB.lower[1],
            height=0.002,
            color=pbu.RGBA(0.0, 0.0, 0, 1.0),
            collision=True,
            client=client,
        )

        pos = copy.deepcopy(GOAL_POSITIONS[line_y])
        pos[0] -= 0.1
        if line_y < 2:
            pbu.set_pose(line, pbu.Pose(pbu.Point(*pos)), client=client)

    stool = create_pillar(
        width=STOOL_AABB.upper[0] - STOOL_AABB.lower[0],
        length=STOOL_AABB.upper[1] - STOOL_AABB.lower[1],
        height=0.1,
        color=pbu.TAN,
        collision=True,
        client=client,
    )

    surface = create_pillar(
        width=FLOOR_WIDTH + 2 * IK_RADIUS,
        length=FLOOR_LENGTH,
        height=0.01,
        color=pbu.RGBA(0, 0, 0, 0),
        collision=True,
        client=client,
    )
    client.changeDynamics(surface, -1, lateralFriction=FRICTION_MID)

    ik_region = create_pillar(
        width=IK_AABB.upper[0] - IK_AABB.lower[0],
        length=IK_AABB.upper[1] - IK_AABB.lower[1],
        height=0.002,
        color=pbu.TAN,
        collision=True,
        client=client,
    )

    client.changeDynamics(ik_region, -1, lateralFriction=FRICTION_MID)

    pbu.set_pose(floor, pbu.Pose(pbu.Point(*FLOOR_POS)), client=client)
    pbu.set_pose(ik_region, pbu.Pose(pbu.Point(*IK_POS)), client=client)
    pbu.set_pose(
        surface,
        pbu.Pose(pbu.Point(x=FLOOR_POS[0] - IK_RADIUS, y=FLOOR_POS[1], z=0.05)),
        client=client,
    )

    GROUP_GHOST = 1
    GROUP_FLOOR_ROBOT = 2

    GHOST_MASK = GROUP_FLOOR_ROBOT
    FLOOR_ROBOT_MASK = GROUP_FLOOR_ROBOT | GROUP_GHOST

    client.setCollisionFilterGroupMask(floor, -1, GROUP_FLOOR_ROBOT, FLOOR_ROBOT_MASK)
    client.setCollisionFilterGroupMask(
        robot.body, -1, GROUP_FLOOR_ROBOT, FLOOR_ROBOT_MASK
    )

    pucks = []
    if friction is not None:
        puck = create_puck(friction, client)
        client.setCollisionFilterGroupMask(puck, -1, GROUP_GHOST, GHOST_MASK)
        pbu.set_pose(puck, pose, client=client)
        pucks.append(puck)

    while len(pucks) < PUCK_PARTICLES:
        friction_sample = 1.0 / random.uniform(
            1.0 / FRICTION_LOWER, 1.0 / FRICTION_UPPER
        )
        puck = create_puck(friction_sample, client, alpha=0.2)
        client.setCollisionFilterGroupMask(puck, -1, GROUP_GHOST, GHOST_MASK)

        pbu.set_pose(puck, pose, client=client)
        pucks.append(puck)

    # Open the fingers
    pbu.set_joint_positions(
        robot,
        robot.get_group_joints(GRIPPER_GROUP, client=client),
        (np.array(OPEN_GRIPPER_POS) + np.array(CLOSE_GRIPPER_POS)) / 2.0,
        client=client,
    )

    pbu.set_joint_positions(
        robot,
        robot.get_group_joints(ARM_GROUP, client=client),
        DEFAULT_ARM_POS,
        client=client,
    )

    world = World(
        client_id=client_id, robot=robot, environment=[], floor=floor, objects=pucks
    )
    return world


def follow_joint_path(
    world: World, joint_path: List[List[float]], lead_step: float
) -> List[pbu.Pose]:
    if len(joint_path) > 0:
        pbu.set_joint_positions(
            world.robot,
            world.robot.get_group_joints(ARM_GROUP, client=world.client),
            joint_path[0],
            client=world.client,
        )
        controller = follow_path(
            world.robot,
            world.robot.get_group_joints(ARM_GROUP, client=world.client),
            joint_path,
            lead_step=lead_step,
            max_force=None,
            client=world.client,
        )
        for _ in iter(pbu.simulate_controller(controller, client=world.client)):
            if pbu.has_gui(client=world.client):
                time.sleep(0.0001)

        for _ in range(2000):
            world.client.stepSimulation()
            if pbu.has_gui(client=world.client):
                time.sleep(0.0001)

    puck_poses = []
    for puck in world.objects:
        puck_poses.append(pbu.get_pose(puck, client=world.client))

    return puck_poses


SHOOT_CONTROLLER_PC = 5
DEFAULT_EFFORT = 0.02


def shoot_controller(
    state,
    world,
    pucks,
    pose,
    target_point,
    pc_steps=SHOOT_CONTROLLER_PC,
    effort=DEFAULT_EFFORT,
) -> Tuple[List[pbu.PoseType], List[List[float]], float]:
    arm_joints = world.robot.get_group_joints(ARM_GROUP, client=world.client)

    DIST_MULT = 0.05
    # Calculate pre push conf
    world_T_oriented = pbu.multiply(
        pbu.Pose(euler=pbu.Euler(pitch=math.pi / 2.0)),
        pbu.Pose(euler=pbu.Euler(roll=-math.pi)),
    )

    # dot = create_cylinder(0.01, 1.0, 0, (1, 0, 0, 1), client=world.client, collision=False)
    # set_pose(dot, Pose(Point(x=target_point[0], y=target_point[1])), client=world.client)

    point_vec = np.array([target_point[0] - pose[0][0], target_point[1] - pose[0][1]])
    point_vec = point_vec / np.linalg.norm(point_vec)

    oriented_T_centered = pbu.Pose(
        euler=pbu.Euler(roll=-math.atan2(point_vec[1], point_vec[0]))
    )
    centered_T_final = pbu.Pose(pbu.Point(x=-0.02, y=0, z=0.07))
    puck_center = pbu.Pose(point=pose[0])
    world_T_final = pbu.multiply(
        pbu.multiply(pbu.multiply(puck_center, world_T_oriented), oriented_T_centered),
        centered_T_final,
    )
    world_T_pushed = pbu.multiply(
        pbu.Pose(pbu.Point(x=point_vec[0] * DIST_MULT, y=point_vec[1] * DIST_MULT)),
        world_T_final,
    )

    KP = 0.01
    with pbu.LockRenderer(client=world.client, lock=True):
        if state is not None:
            current_q = state.current_q

        joint_path = plan_workspace_motion(
            world.robot,
            [world_T_final, world_T_pushed],
            obstacles=[],
            client=world.client,
            resolutions=0.2,
        )
        if joint_path is None:
            return None, [], None

        if state is not None:
            motion_plan = plan_motion(
                world,
                current_q,
                joint_path[0],
                obstacles=[world.floor],
            )
            if motion_plan is None:
                return None, [], None

        else:
            pbu.set_joint_positions(
                world.robot, arm_joints, joint_path[0], client=world.client
            )

    if state is not None:
        state.apply_sequence([motion_plan])

    for step in range(pc_steps):
        # Render only the last one
        with pbu.LockRenderer(client=world.client, lock=(step + 1) < pc_steps):
            pbu.set_joint_positions(
                world.robot, arm_joints, joint_path[0], client=world.client
            )

            for puck in pucks:
                pbu.set_pose(puck, pose, client=world.client)

            puck_poses = follow_joint_path(world, joint_path, lead_step=effort)

            # Get mean puck pose
            mean_x = np.median([pp[0][0] for pp in puck_poses])
            mean_y = np.median([pp[0][1] for pp in puck_poses])
            delta_p = np.array([mean_x, mean_y]) - np.array(target_point)
            effort = effort - delta_p[0] * KP

    # remove_body(dot, client=world.client)
    return puck_poses, joint_path, effort


def push_execute_fn(
    action: Action,
    belief: PuckSlideBelief,
    state: PuckState,
    store: AliasStore,
) -> Tuple[PuckState, PuckObservation]:
    obj, region = action.args

    puck_pose = belief.puck_pose

    update_sim(
        state.world.objects[1:],
        puck_pose,
        frictions=belief.particle_frictions,
        client=state.world.client,
    )

    tp = REGION_POSES[store.get(region)][0][:2]
    new_poses, joint_path, effort = shoot_controller(
        state, state.world, state.world.objects, puck_pose, tp
    )

    if new_poses is None:
        return state, PuckObservation(puck_pose=puck_pose)

    for obj in state.world.objects:
        pbu.set_pose(obj, new_poses[0], state.world.client)

    return state, PuckObservation(
        puck_pose=new_poses[0], joint_path=joint_path, effort=effort
    )


def effects_fn(
    puck: str,
    target_point: List[float],
    belief: PuckSlideBelief,
    pc_steps: int,
    effort: float,
    store: AliasStore,
) -> AbstractBeliefSet:
    pbu.set_pose(store.get(puck), belief.puck_pose, client=belief.world.client)
    belief.update_sim()

    new_poses, _, _ = shoot_controller(
        None,
        belief.world,
        belief.world.objects,
        belief.puck_pose,
        target_point,
        pc_steps=pc_steps,
        effort=effort,
    )
    if new_poses is None:
        ab = belief.abstract(store)
        return AbstractBeliefSet(
            ab_counts={ab: len(belief.particle_frictions)},
            belief_map={ab: [copy.deepcopy(belief)]},
        )

    updated_beliefs = []
    for new_pose in new_poses:
        new_belief = copy.deepcopy(belief)
        new_belief.update_particle_frictions(new_poses, new_pose)
        new_belief.reweight_particles()
        new_belief.puck_pose = new_pose
        updated_beliefs.append(new_belief)

    return AbstractBeliefSet.from_beliefs(updated_beliefs, store)


def push_effects_fn(
    action: Action, belief: PuckSlideBelief, store: AliasStore
) -> AbstractBeliefSet:
    puck, region = action.args
    tp = REGION_POSES[store.get(region)][0][:2]
    return effects_fn(
        puck,
        tp,
        belief,
        pc_steps=SHOOT_CONTROLLER_PC,
        effort=DEFAULT_EFFORT,
        store=store,
    )


def nudge_execute_fn(
    action: Action,
    belief: PuckSlideBelief,
    state: PuckState,
    store: AliasStore,
) -> Tuple[PuckState, PuckObservation]:
    puck, nd = action.args
    puck_pose = belief.puck_pose
    direction = store.get(nd)

    tp = REGION_POSES[store.get(f"{OBJ}ik")][0][:2]
    tp[0] += direction[0]
    tp[1] += direction[1]

    update_sim(
        state.world.objects[1:],
        puck_pose,
        frictions=belief.particle_frictions,
        client=state.world.client,
    )

    new_poses, joint_path, effort = shoot_controller(
        state,
        state.world,
        state.world.objects,
        puck_pose,
        tp,
        pc_steps=5,
        effort=0.005,
    )
    if new_poses is None:
        return state, PuckObservation(puck_pose=puck_pose)

    for obj in state.world.objects:
        pbu.set_pose(obj, new_poses[0], state.world.client)

    return state, PuckObservation(
        puck_pose=new_poses[0], joint_path=joint_path, effort=effort
    )


def nudge_effects_fn(
    action: Action, belief: PuckSlideBelief, store: AliasStore
) -> AbstractBeliefSet:
    (puck, nd) = action.args
    direction = store.get(nd)
    tp = REGION_POSES[store.get(f"{OBJ}ik")][0][:2]
    tp[0] += direction[0]
    tp[1] += direction[1]
    r = effects_fn(puck, tp, belief, pc_steps=5, effort=0.005, store=store)
    return r


def sample_nudge_direction(args, store):
    direction = np.array([np.random.normal(0, 0.02), np.random.normal(0, 0.3)])
    direction = direction / np.linalg.norm(direction) * 0.2
    return direction


def test_in(pose, region):
    world_T_obj = pose
    world_T_region = REGION_POSES[region]
    region_aabb = SHAPES[region]

    for object_vert in pbu.get_oobb_vertices(
        pbu.OOBB(
            pbu.AABB(lower=[-0.01, -0.01, -0.01], upper=[0.01, 0.01, 0.01]), world_T_obj
        )
    ):
        if pbu.oobb_contains_point(object_vert, pbu.OOBB(region_aabb, world_T_region)):
            return True
    return False


class PuckSlideEnv(TampuraEnv):
    def __init__(self, *args, **kwargs):
        super(PuckSlideEnv, self).__init__(*args, **kwargs)

    def initialize(self) -> Tuple[PuckSlideBelief, AliasStore]:
        store = AliasStore()
        state_friction = random.uniform(FRICTION_LOWER, FRICTION_UPPER)
        world = setup_puck_world(
            pose=PUCK_START_POSE, friction=state_friction, vis=self.vis
        )
        sim_world = setup_puck_world(pose=PUCK_START_POSE, friction=None, vis=False)
        self.state = PuckState(world)

        # Create initial object set with pointers to simulation items here
        store.set(f"{OBJ}puck", sim_world.objects[0], "physical")
        for region in REGIONS:
            store.set(region, region, "region")

        store.certified += [Atom("is-goal", [r]) for r in GOAL_OBJECTS]

        obs = PuckObservation(puck_pose=PUCK_START_POSE)

        return PuckSlideBelief(sim_world, obs, store), store

    def get_problem_spec(self) -> ProblemSpec:
        predicates = [
            Predicate("in", ["physical", "region"]),
            Predicate("known-friction", ["physical"]),
            Predicate("reachable", ["region"]),
            Predicate("off-table", []),
            Predicate("target-point", ["region", "point"]),
            Predicate("is-goal", ["region"]),
        ]

        stream_schemas = [
            StreamSchema(
                name="nudge-direction",
                output="?d",
                output_type="direction",
                sample_fn=sample_nudge_direction,
            )
        ]

        action_schemas = [
            ActionSchema(
                name="push-to",
                inputs=["?o", "?r"],
                depends=[Atom("known-friction", ["?o"])],
                input_types=["physical", "region"],
                preconditions=[Atom("in", ["?o", f"{OBJ}ik"]), Atom("is-goal", ["?r"])],
                effects=[Not(Atom("in", ["?o", f"{OBJ}ik"]))],
                verify_effects=[
                    Atom("in", ["?o", "?r"]),
                    Atom("known-friction", ["?o"]),
                ],
                effects_fn=push_effects_fn,
                execute_fn=push_execute_fn,
            ),
            ActionSchema(
                name="nudge",
                inputs=["?o", "?d"],
                input_types=["physical", "direction"],
                preconditions=[Atom("in", ["?o", f"{OBJ}ik"])],
                verify_effects=[Atom("known-friction", ["?o"])],
                effects_fn=nudge_effects_fn,
                execute_fn=nudge_execute_fn,
            ),
            NoOp(),
        ]

        reward = Atom("in", [f"{OBJ}puck", f"{OBJ}goal0"])

        spec = ProblemSpec(
            predicates=predicates,
            action_schemas=action_schemas,
            stream_schemas=stream_schemas,
            reward=reward,
        )

        return spec


register_env("puck_slide", PuckSlideEnv)
