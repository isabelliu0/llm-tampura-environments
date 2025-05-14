import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
from tampura.config.config import register_env
from tampura.environment import TampuraEnv
from tampura.spec import ProblemSpec
from tampura.structs import (AbstractBelief, ActionSchema, AliasStore, Belief,
                             NoOp, Predicate, State, effect_from_execute_fn)
from tampura.symbolic import OBJ, Atom, ForAll

PICK_ONE_SUCCESS = 0.8
PICK_BOTH_SUCCESS = 0.4
OBJECTS = [f"{OBJ}o1", f"{OBJ}o2"]


@dataclass
class HoldingObservation:
    holding: List[str] = field(default_factory=lambda: [])


class HoldingBelief(Belief):
    def __init__(self, holding=[]):
        self.holding = holding

    def update(self, a, o, s):
        return HoldingBelief(holding=o.holding)

    def abstract(self, store: AliasStore):
        return AbstractBelief([Atom("holding", [o]) for o in self.holding])

    def vectorize(self):
        return np.array([int(obj in self.holding) for obj in OBJECTS])


def pick_execute_fn(a, b, s, store):
    holding = (
        list(set(b.holding + list(a.args)))
        if random.random() < PICK_ONE_SUCCESS
        else b.holding
    )
    return State(), HoldingObservation(holding)


def pick_both_execute_fn(a, b, s, store):
    holding = (
        list(set(b.holding + list(a.args)))
        if random.random() < PICK_BOTH_SUCCESS
        else b.holding
    )
    return State(), HoldingObservation(holding)


class ToyDiscrete(TampuraEnv):
    def initialize(self):
        store = AliasStore()
        for o in OBJECTS:
            store.set(o, o, "physical")

        return HoldingBelief(), store

    def get_problem_spec(self) -> ProblemSpec:
        predicates = [
            Predicate("holding", ["physical"]),
        ]

        action_schemas = [
            ActionSchema(
                name="pick",
                inputs=["?o1"],
                input_types=["physical"],
                verify_effects=[Atom("holding", ["?o1"])],
                execute_fn=pick_execute_fn,
                effects_fn=effect_from_execute_fn(pick_execute_fn),
            ),
            ActionSchema(
                name="pick-both",
                inputs=["?o1", "?o2"],
                input_types=["physical", "physical"],
                verify_effects=[Atom("holding", ["?o1"]), Atom("holding", ["?o2"])],
                execute_fn=pick_both_execute_fn,
                effects_fn=effect_from_execute_fn(pick_both_execute_fn),
            ),
            NoOp(),
        ]

        reward = ForAll(Atom("holding", ["?o"]), ["?o"], ["physical"])

        spec = ProblemSpec(
            predicates=predicates,
            action_schemas=action_schemas,
            reward=reward,
        )

        return spec


register_env("toy_discrete", ToyDiscrete)
