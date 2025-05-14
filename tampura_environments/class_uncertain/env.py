from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from tampura.config.config import register_env
from tampura.environment import TampuraEnv
from tampura.spec import ProblemSpec
from tampura.structs import (AbstractBelief, AbstractBeliefSet, Action,
                             ActionSchema, AliasStore, Atom, Belief, Exists,
                             NoOp, Observation, Predicate, StreamSchema,
                             effect_from_execute_fn)
from tampura.symbolic import And, Expr, ForAll, Imply, Not

import tampura_environments.panda_utils.pb_utils as pbu
from tampura_environments.panda_utils.panda_env_utils import (
    ARM_GROUP, CLIENT_MAP, EXCLUDE_CLASSES, GRIPPER_GROUP, OPEN_GRIPPER_POS,
    PandaRobot, SceneState, World, create_default_env, create_ycb, get_grasp,
    grasp_attachment, grasp_ik, ik, pick_execute, place_execute, plan_motion,
    plan_workspace_motion, setup_robot_pybullet)
from tampura_environments.panda_utils.robot import DEFAULT_ARM_POS

FRUIT_CLASSES = ["apple", "lemon", "strawberry"]
BALL_CLASSES = ["tennis_ball", "racquetball", "baseball"]

# For risk sensitivity experiments
# DROPABILITY = {cl: 0.0 for cl in FRUIT_CLASSES} | {cl: 0.35 for cl in BALL_CLASSES}

# For faster runtime
DROPABILITY = {cl: 0.0 for cl in FRUIT_CLASSES} | {cl: 0.85 for cl in BALL_CLASSES}

ALL_CLASSES = FRUIT_CLASSES + BALL_CLASSES
BOWL_CLASS = "bowl"
KNOWN_CLASS_THRESH = 0.9


class ClassCategoryBelief(Belief):
    def __init__(self, world: World, *args, **kwargs):
        super(ClassCategoryBelief, self).__init__(*args, **kwargs)
        self.world = world

        # Start with a uniform distribution
        self.class_distributions = {
            obj: [1.0 / len(ALL_CLASSES)] * len(ALL_CLASSES)
            for obj in self.world.objects
        }
        self.grasp = None
        self.grasp_object = None
        self.broken = []
        self.ons = {obj: self.world.regions[0] for obj in self.world.objects}

    def vectorize(self):
        """For RL."""
        all_obj_props = []
        for obj in self.world.objects:
            ons_vec = [
                int(obj in self.ons and r == self.ons[obj]) for r in self.world.regions
            ]
            all_obj_props.append(np.array(self.class_distributions[obj] + ons_vec))
        return np.concatenate(
            all_obj_props
            + [np.array([int(o in self.broken) for o in self.world.objects])]
        )

    def update(
        self,
        action: Action,
        observation: ClassUncertainObservation,
        store: AliasStore,
    ) -> Belief:
        new_belief = copy.deepcopy(self)
        if observation is None:
            return new_belief

        if observation.broken is not None:
            new_belief.broken.append(observation.broken)

        new_belief.grasp = observation.grasp
        new_belief.grasp_object = observation.grasp_object

        if observation.grasp is not None:
            new_belief.ons = {
                obj: reg
                for obj, reg in self.ons.items()
                if obj != observation.grasp_object
            }
        elif action.name == "drop":
            new_belief.ons[action.args[0]] = action.args[2]

        if observation.detected_object is not None:
            new_belief.class_distributions[observation.detected_object] = [
                0 for _ in range(len(ALL_CLASSES))
            ]
            new_belief.class_distributions[observation.detected_object][
                ALL_CLASSES.index(observation.class_label)
            ] = 1.0

        return new_belief

    def abstract(self, store: AliasStore) -> AbstractBelief:
        known: List[Expr] = []

        for obj in self.world.objects:
            if max(self.class_distributions[obj]) > KNOWN_CLASS_THRESH:
                class_index = np.argmax(self.class_distributions[obj])
                if ALL_CLASSES[class_index] in FRUIT_CLASSES:
                    known.append(Atom(f"is-fruit", [obj]))
            for fruit_class in FRUIT_CLASSES:
                if self.class_distributions[obj][ALL_CLASSES.index(fruit_class)] > 0:
                    known.append(Atom(f"possible-fruit", [obj]))
                    break

        if self.grasp is not None:
            known.append(Atom("at-grasp", [self.grasp_object, self.grasp]))

        for on_obj, on_region in self.ons.items():
            known.append(Atom("on", [on_obj, on_region]))

        for broken in self.broken:
            known.append(Atom("broken", [broken]))

        return AbstractBelief(known)


@dataclass
class ClassUncertainObservation(Observation):
    detected_object: Optional[str] = None
    class_label: Optional[str] = None
    grasp: Optional[str] = None
    grasp_object: Optional[str] = None
    broken: Optional[str] = None


def inspect_execute_fn(
    action: Action,
    belief: ClassCategoryBelief,
    state: SceneState,
    store: AliasStore,
) -> AbstractBeliefSet:
    (o,) = action.args
    if state is None:
        class_obs = np.random.choice(ALL_CLASSES, p=belief.class_distributions[o])
    else:

        # Move robot to inspection pose
        obj_point = pbu.get_pose(store.get(o), client=belief.world.client)[0]
        inspect_gripper_pose = pbu.multiply(
            pbu.Pose(pbu.Point(z=0.1)),
            pbu.Pose(obj_point, euler=pbu.Euler(pitch=np.pi / 2)),
        )

        pre_confs = plan_workspace_motion(
            belief.world.robot,
            [inspect_gripper_pose],
            obstacles=[],
            client=belief.world.client,
        )

        if pre_confs is not None:
            motion_plan = plan_motion(
                belief.world,
                state.current_q,
                pre_confs[0],
                obstacles=[belief.world.floor] + list(set(belief.world.environment)),
            )
            state.apply_sequence([motion_plan], teleport=False)

        class_obs = state.world.categories[state.world.objects.index(store.get(o))]

    return state, ClassUncertainObservation(detected_object=o, class_label=class_obs)


def pick_execute_fn(
    action: Action,
    belief: ClassCategoryBelief,
    state: SceneState,
    store: AliasStore,
) -> Tuple[SceneState, ClassUncertainObservation]:
    o, g, r = action.args
    pre_confs = grasp_ik(
        belief.world,
        store.get(o),
        pbu.get_pose(store.get(o), client=belief.world.client),
        store.get(g),
        # obstacles=[belief.world.floor] #+ list(set(belief.world.environment)),
    )

    if pre_confs is not None:
        if state is None:
            sampled_class = np.random.choice(
                ALL_CLASSES, p=belief.class_distributions[o]
            )
        else:
            sampled_class = state.world.categories[
                state.world.objects.index(store.get(o))
            ]
            motion_plan = plan_motion(
                belief.world,
                state.current_q,
                pre_confs[0],
                # obstacles=[belief.world.floor] #+ list(set(belief.world.environment)),
            )
            if motion_plan is not None:
                state = pick_execute(
                    state,
                    store.get(g),
                    motion_plan,
                    pre_confs,
                    full_close=False,
                )
            else:
                logging.debug("Motion plan is None")
                return state, ClassUncertainObservation()

    else:
        logging.debug("Ik sol is none")
        return state, ClassUncertainObservation()

    if random.random() < DROPABILITY[sampled_class]:
        return state, ClassUncertainObservation(broken=o)
    else:
        return state, ClassUncertainObservation(grasp=g, grasp_object=o)


def place_execute_fn(
    action: Action, belief: ClassCategoryBelief, state: SceneState, store: AliasStore
) -> Tuple[SceneState, ClassUncertainObservation]:
    o, g, r = action.args
    obs = ClassUncertainObservation()
    place_pose = pbu.multiply(
        pbu.Pose(pbu.Point(z=0.1)),
        pbu.get_pose(store.get(r), client=belief.world.client),
    )
    conf = ik(belief.world, store.get(o), place_pose, store.get(g))

    # TODO: implement collision checking

    if conf is not None:
        if state is not None:
            motion_plan = plan_motion(
                belief.world,
                state.current_q,
                conf,
                obstacles=[
                    state.world.floor
                ],  # + list(set(state.world.environment) - set([store.get(o)])),
                attachments=[grasp_attachment(belief.world, store.get(g))],
            )

            if motion_plan is not None:
                state = place_execute(state, motion_plan)
                # sim a few steps to let the object settle
                for _ in range(200):
                    state.world.client.stepSimulation()
                    time.sleep(0.01)

        return state, obs
    else:
        obs = ClassUncertainObservation()
        obs.grasp = belief.grasp
        obs.grasp_object = belief.grasp_object
        return None, obs


def grasp_sample_fn_wrapper(world):
    def grasp_sample_fn(args: List[str], store: AliasStore):
        (obj,) = store.get_all(args)
        g = get_grasp(
            world,
            obj,
            obstacles=world.environment,
            client=world.client,
            grasp_mode="saved",
        )
        return g

    return grasp_sample_fn


def setup_world(scene_data, gui=False, **kwargs) -> World:
    """Setup the world to have a set of YCB objects at certain poses defined in
    the scene_data dictionary."""
    robot_body, client = setup_robot_pybullet(gui=gui)

    robot = PandaRobot(robot_body, client=client)

    floor, obstacles = create_default_env(client=client, **kwargs)

    movable = []
    regions = [floor]

    all_objects = []

    for category, scale in zip(scene_data["categories"], scene_data["scales"]):
        obj = create_ycb(
            category, client=client, use_concave=True, scale=scale, mass=0.1
        )
        logging.debug(f"Created object {category} with body {obj}")

        if category == BOWL_CLASS:
            regions.append(obj)
        else:
            movable.append(obj)

        all_objects.append(obj)

    client_id = len(CLIENT_MAP)
    CLIENT_MAP[client_id] = client

    for obj, pose in zip(all_objects, scene_data["poses"]):
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

    client.setGravity(0, 0, -10)

    for _ in range(2000):
        client.stepSimulation()

    return World(
        client_id=client_id,
        robot=robot,
        environment=obstacles + movable,
        floor=floor,
        objects=movable,
        categories=scene_data["categories"],
        regions=regions,
    )


class ClassUncertainEnv(TampuraEnv):
    def __init__(self, *args, **kwargs):
        super(ClassUncertainEnv, self).__init__(*args, **kwargs)

    def get_scene_data(self):
        dataset_dir = "tampura_environments/class_uncertain/problems"
        world_json_file = random.choice(os.listdir(dataset_dir))

        logging.info("Loading scene from {}".format(world_json_file))

        # Load json from file
        with open(os.path.join(dataset_dir, world_json_file)) as f:
            scene_data_json = json.load(f)

        return scene_data_json

    def initialize(self) -> Tuple[ClassCategoryBelief, AliasStore]:
        store = AliasStore()
        self.scene_data = None
        while self.scene_data is None:
            scene_data_json = self.get_scene_data()
            if set(EXCLUDE_CLASSES).isdisjoint(set(scene_data_json["categories"])):
                self.scene_data = scene_data_json
                world = setup_world(self.scene_data, gui=self.vis)
                self.state = SceneState(world)
                self.sim_world = setup_world(self.scene_data, gui=False)

        self.sim_world.objects = [
            store.add_typed(o, "physical") for o in self.sim_world.objects
        ]
        self.sim_world.regions = [
            store.add_typed(o, "region") for o in self.sim_world.regions
        ]
        store.certified.append(Atom("is-bowl", [self.sim_world.regions[1]]))

        return ClassCategoryBelief(self.sim_world), store

    def get_problem_spec(self):
        predicates = [
            Predicate("at-grasp", ["physical", "grasp"]),
            Predicate("possible-fruit", ["physical"]),
            Predicate("on", ["physical", "region"]),
            Predicate("broken", ["physical"]),
            Predicate("is-fruit", ["physical"]),
            Predicate("is-bowl", ["region"]),
            Predicate("object-grasp", ["physical", "grasp"]),
        ]

        not_holding = Not(
            Exists(
                Atom("at-grasp", ["?o2", "?g2"]), ["?o2", "?g2"], ["physical", "grasp"]
            )
        )
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
        ]
        action_schemas = [
            ActionSchema(
                name="pick",
                inputs=["?o", "?g", "?r"],
                depends=[Atom("is-fruit", ["?o"])],
                input_types=["physical", "grasp", "region"],
                preconditions=[
                    not_holding,
                    Atom("object-grasp", ["?o", "?g"]),
                    Not(Atom("broken", ["?o"])),
                    Atom("on", ["?o", "?r"]),
                ],
                verify_effects=[
                    Atom("at-grasp", ["?o", "?g"]),
                    Atom("broken", ["?o"]),
                    Atom("on", ["?o", "?r"]),
                ],
                effects_fn=effect_from_execute_fn(pick_execute_fn),
                execute_fn=pick_execute_fn,
            ),
            ActionSchema(
                name="drop",
                inputs=["?o", "?g", "?r"],
                input_types=["physical", "grasp", "region"],
                preconditions=[
                    Atom("is-bowl", ["?r"]),
                    Not(Atom("broken", ["?o"])),
                    Atom("at-grasp", ["?o", "?g"]),
                ],
                effects=[Not(Atom("at-grasp", ["?o", "?g"])), Atom("on", ["?o", "?r"])],
                effects_fn=effect_from_execute_fn(place_execute_fn),
                execute_fn=place_execute_fn,
            ),
            ActionSchema(
                name="inspect",
                inputs=["?o"],
                input_types=["physical"],
                preconditions=[
                    not_holding,
                    Atom("possible-fruit", ["?o"]),
                    Not(Atom("broken", ["?o"])),
                ],
                verify_effects=[
                    Atom("is-fruit", ["?o"]),
                    Atom("possible-fruit", ["?o"]),
                ],
                effects_fn=effect_from_execute_fn(inspect_execute_fn),
                execute_fn=inspect_execute_fn,
            ),
            NoOp(),
        ]

        in_bowl = Exists(
            And([Atom("on", ["?o", "?r"]), Atom("is-bowl", ["?r"])]), ["?r"], ["region"]
        )
        fruit_in_bowl = Imply(Atom("possible-fruit", ["?o"]), in_bowl)
        reward = ForAll(
            And([fruit_in_bowl, Not(Atom("broken", ["?o"]))]), ["?o"], ["physical"]
        )

        spec = ProblemSpec(
            predicates=predicates,
            action_schemas=action_schemas,
            reward=reward,
            stream_schemas=stream_schemas,
        )

        return spec


register_env("class_uncertain", ClassUncertainEnv)
