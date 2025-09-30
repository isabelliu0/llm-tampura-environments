"""Trajectory logger for collecting training data for LLM-based planning."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tampura.structs import AbstractBelief, Action, AliasStore, Belief, Observation
from tampura.symbolic import Atom


@dataclass
class TrajectoryData:
    """Single timestep of trajectory data."""

    timestep: int
    domain_file: Optional[str] = None
    problem_file: Optional[str] = None
    abstract_belief: List[str] = field(default_factory=list)
    observation: Dict[str, Any] = field(default_factory=dict)
    action_taken: Optional[Dict[str, Any]] = None
    reward: float = 0.0
    next_abstract_belief: List[str] = field(default_factory=list)
    certified_facts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrajectoryData:
        """Create from dictionary."""
        return cls(**data)


class TrajectoryLogger:
    """Logger for collecting trajectory data during TAMPURA execution."""

    def __init__(self, save_dir: str, task_name: str, enabled: bool = True):
        """Initialize trajectory logger."""
        self.enabled = enabled
        if not self.enabled:
            return

        self.save_dir = Path(save_dir)
        self.task_name = task_name
        self.task_dir = self.save_dir / task_name
        self.pddl_dir = self.task_dir / "pddl_files"
        self.trajectory_file = self.task_dir / "trajectory.json"

        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.pddl_dir.mkdir(exist_ok=True)

        self.trajectory: List[TrajectoryData] = []
        self.current_timestep = 0
        self.metadata: Dict[str, Any] = {
            "task_name": task_name,
            "total_timesteps": 0,
            "total_reward": 0.0,
            "success": False,
        }

        logging.info(f"[TrajectoryLogger] Initialized for task '{task_name}' at {self.task_dir}")

    def log_timestep(
        self,
        abstract_belief: AbstractBelief,
        observation: Optional[Observation],
        action: Optional[Action],
        reward: float,
        store: AliasStore,
        next_abstract_belief: Optional[AbstractBelief] = None,
        domain_file: Optional[str] = None,
        problem_file: Optional[str] = None,
    ) -> None:
        """Log a single timestep of trajectory data.

        Args:
            abstract_belief: Current abstract belief state
            observation: Current observation
            action: Action taken at this timestep
            reward: Reward received
            store: Alias store with certified facts
            next_abstract_belief: Resulting abstract belief after action
            domain_file: Path to PDDL domain file (if available)
            problem_file: Path to PDDL problem file (if available)
        """
        if not self.enabled:
            return

        ab_strings = [self._atom_to_string(atom) for atom in abstract_belief.items]
        obs_dict = self._observation_to_dict(observation) if observation else {}
        action_dict = None
        if action is not None:
            action_dict = {
                "name": action.name,
                "args": list(action.args) if action.args else [],
            }
        next_ab_strings = []
        if next_abstract_belief is not None:
            next_ab_strings = [self._atom_to_string(atom) for atom in next_abstract_belief.items]
        certified_strings = [self._atom_to_string(atom) for atom in store.certified]

        saved_domain_file = None
        saved_problem_file = None
        if domain_file and os.path.exists(domain_file):
            saved_domain_file = str(self.pddl_dir / f"timestep_{self.current_timestep}_domain.pddl")
            shutil.copy2(domain_file, saved_domain_file)
        if problem_file and os.path.exists(problem_file):
            saved_problem_file = str(self.pddl_dir / f"timestep_{self.current_timestep}_problem.pddl")
            shutil.copy2(problem_file, saved_problem_file)

        data = TrajectoryData(
            timestep=self.current_timestep,
            domain_file=saved_domain_file,
            problem_file=saved_problem_file,
            abstract_belief=ab_strings,
            observation=obs_dict,
            action_taken=action_dict,
            reward=reward,
            next_abstract_belief=next_ab_strings,
            certified_facts=certified_strings,
        )
        self.trajectory.append(data)
        self.current_timestep += 1
        logging.debug(f"[TrajectoryLogger] Logged timestep {self.current_timestep - 1}")

    def save(self, success: bool = False, total_reward: float = 0.0) -> str:
        """Save trajectory data to disk."""
        if not self.enabled:
            return ""

        self.metadata["total_timesteps"] = len(self.trajectory)
        self.metadata["success"] = success
        self.metadata["total_reward"] = total_reward

        trajectory_dicts = [data.to_dict() for data in self.trajectory]
        full_data = {
            "metadata": self.metadata,
            "trajectory": trajectory_dicts,
        }

        with open(self.trajectory_file, "w") as f:
            json.dump(full_data, f, indent=2)
        logging.info(
            f"[TrajectoryLogger] Saved {len(self.trajectory)} timesteps to {self.trajectory_file}"
        )
        return str(self.trajectory_file)

    def load(self, trajectory_file: str) -> List[TrajectoryData]:
        """Load trajectory data from file."""
        with open(trajectory_file, "r") as f:
            full_data = json.load(f)
        self.metadata = full_data.get("metadata", {})
        trajectory_dicts = full_data.get("trajectory", [])
        self.trajectory = [TrajectoryData.from_dict(d) for d in trajectory_dicts]
        logging.info(f"[TrajectoryLogger] Loaded {len(self.trajectory)} timesteps from {trajectory_file}")
        return self.trajectory

    def _atom_to_string(self, atom: Atom) -> str:
        """Convert an Atom to a string representation."""
        if not hasattr(atom, 'pred_name'):
            return str(atom)

        if not atom.args or len(atom.args) == 0:
            return f"({atom.pred_name})"
        else:
            args_str = " ".join(str(arg) for arg in atom.args)
            return f"({atom.pred_name} {args_str})"

    def _observation_to_dict(self, observation: Observation) -> Dict[str, Any]:
        """Convert observation (specific to SLAM2D) to dictionary."""
        obs_dict: Dict[str, Any] = {}

        if hasattr(observation, "regions_in"):
            obs_dict["regions_in"] = list(observation.regions_in)

        if hasattr(observation, "holding"):
            if observation.holding is not None:
                if hasattr(observation.holding, "target"):
                    obs_dict["holding"] = observation.holding.target
                else:
                    obs_dict["holding"] = str(observation.holding)
            else:
                obs_dict["holding"] = None

        if hasattr(observation, "robot_pose"):
            if observation.robot_pose is not None:
                obs_dict["robot_pose"] = {
                    "x": observation.robot_pose.x,
                    "y": observation.robot_pose.y,
                    "theta": observation.robot_pose.theta,
                }
            else:
                obs_dict["robot_pose"] = None

        if hasattr(observation, "collision"):
            obs_dict["collision"] = observation.collision

        return obs_dict

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged trajectory."""
        if not self.enabled or len(self.trajectory) == 0:
            return {}

        total_reward = sum(data.reward for data in self.trajectory)
        actions_taken = [data.action_taken for data in self.trajectory if data.action_taken]

        action_counts: Dict[str, int] = {}
        for action_dict in actions_taken:
            if action_dict:
                name = action_dict["name"]
                action_counts[name] = action_counts.get(name, 0) + 1

        return {
            "task_name": self.task_name,
            "total_timesteps": len(self.trajectory),
            "total_reward": total_reward,
            "unique_actions": len(action_counts),
            "action_distribution": action_counts,
            "has_pddl_files": any(data.domain_file is not None for data in self.trajectory),
        }


def create_logger_from_config(config: Dict[str, Any], task_name: str) -> TrajectoryLogger:
    """Create a trajectory logger from a config dictionary."""
    save_dir = config.get("data_save_dir", "./training_data")
    enabled = config.get("collect_data", False)

    return TrajectoryLogger(
        save_dir=save_dir,
        task_name=task_name,
        enabled=enabled,
    )