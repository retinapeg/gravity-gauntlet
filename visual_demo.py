"""Judge-facing, provenance-safe rollout replay for Gravity Gauntlet.

All simulation and collision logic remains in ``gravity_env.py``.  This file
renders exact recorded samples over a freshly reconstructed matching seeded
universe.  Its keyboard-driven development mode is always labelled
``LOCAL PREVIEW`` and is never presented as Daytona execution.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from gravity_env import ACTION_HOLD_STEPS, GravityEnv

try:
    import pygame
except ImportError:  # Allows non-visual tooling to import this module safely.
    pygame = None  # type: ignore[assignment]


WIDTH = 1200
HEIGHT = 800
FPS = 60
TRAIL_LENGTH = 650
REPLAY_TARGET_SECONDS = 8.0
REPLAY_END_HOLD_SECONDS = 2.8
DEFAULT_LIVE_SNAPSHOT_NAME = "gravity-gauntlet-worker-v2"
DEFAULT_LIVE_WORLDS = 8
DEFAULT_LIVE_MAX_STEPS = 500
DEFAULT_LIVE_BASE_SEED = 18_473
DEFAULT_LIVE_RUNS_DIR = Path("runs")
DEFAULT_LIVE_CHECKPOINT_DIR = Path("checkpoints")
LIVE_STATE_SCHEMA_VERSION = 1
LIVE_STATE_POLL_SECONDS = 0.15
LIVE_STATE_FILENAME = "live_state.json"
LIVE_CONTROLLER_MODULE = "src.gravity_gauntlet.demo_controller"
LIVE_TRAINING_PHASES = (
    "READY",
    "FREEZING_POLICY",
    "CREATING_DAYTONA_WORLDS",
    "RUNNING_DAYTONA",
    "COLLECTING_EXPERIENCES",
    "TRAINING_POLICY",
    "POLICY_UPDATED",
    "LAUNCHING_NEXT_GENERATION",
    "GENERATION_COMPLETE",
    "ERROR",
)
LIVE_WORLD_STATES = {
    "CREATING",
    "LIVE",
    "RUNNING",
    "SUCCESS",
    "COLLISION",
    "OUT_OF_BOUNDS",
    "TIMEOUT",
    "RESULT_COLLECTED",
    "ERROR",
}
LIVE_TERMINAL_PHASES = {"GENERATION_COMPLETE", "ERROR"}
LIVE_TRAINER_METRICS = {
    "episodes",
    "transitions",
    "loss",
    "policy_loss",
    "entropy",
    "gradient_norm",
    "parameter_l2_delta",
    "parameter_max_abs_delta",
    "changed_parameter_tensors",
    "changed_parameter_elements",
    "changed_tensors",
    "changed_elements",
    "weights_changed",
    "mean_reward_to_go",
    "std_reward_to_go",
    "advantage_mean",
    "advantage_std",
}

# Preserve the 1200:800 simulation aspect ratio while reserving a fixed right
# column for eight judge-facing universe cards.  This is a rendering transform
# only; recorded coordinates and physics data are never changed.
WORLD_VIEWPORT_X = 90
WORLD_VIEWPORT_Y = 115
WORLD_VIEWPORT_WIDTH = 750
WORLD_VIEWPORT_HEIGHT = 500

SPACE = (3, 5, 17)
WHITE = (232, 242, 255)
MUTED = (126, 151, 187)
CYAN = (69, 225, 255)
GREEN = (85, 255, 174)
RED = (255, 70, 92)

TERMINAL_LIFECYCLE_STATES = {
    "SUCCESS",
    "COLLISION",
    "OUT_OF_BOUNDS",
    "TIMEOUT",
}

PLANET_PALETTE = (
    (80, 186, 255),
    (255, 102, 124),
    (255, 193, 73),
    (151, 102, 255),
    (74, 222, 172),
    (235, 92, 188),
)


@dataclass(frozen=True)
class AttemptTrail:
    """One recorded/local path plus only the metadata present in its artifact."""

    points: tuple[tuple[float, float], ...]
    reward: float | None
    success: bool | None
    seed: int | None = None
    policy_version: int | None = None
    next_policy_version: int | None = None
    generation: int | None = None
    sandbox_id: str | None = None
    trajectory: tuple[dict[str, Any], ...] = ()
    action_vectors: tuple[tuple[float, float], ...] = ()
    universe: dict[str, Any] | None = None
    lifecycle: tuple[dict[str, Any], ...] = ()
    world_index: int | None = None
    termination: str | None = None
    min_clearance: float | None = None
    mean_speed: float | None = None
    max_speed: float | None = None
    generation_average_reward: float | None = None
    generation_best_reward: float | None = None
    generation_success_rate: float | None = None
    generation_collision_rate: float | None = None
    generation_world_count: int | None = None
    provenance: str = "LOCAL PREVIEW"
    daytona_verified: bool = False
    is_champion: bool = False
    champion_declared: bool = False
    source_file: str | None = None


@dataclass(frozen=True)
class LiveTrainingConfig:
    """UI-owned launch settings for one controller generation at a time."""

    repo_root: Path
    snapshot_name: str
    worlds: int
    max_steps: int
    runs_dir: Path
    checkpoint_dir: Path
    live_state_path: Path
    base_seed: int
    start_generation: int = 0
    policy_version: int = 0
    start_checkpoint: Path | None = None


@dataclass(frozen=True)
class LiveWorldSnapshot:
    """One real world entry from the controller's atomic live-state file."""

    world_index: int
    seed: int | None
    sandbox_id: str | None
    lifecycle_state: str | None
    reward: float | None
    success: bool | None
    termination: str | None
    result_collected: bool
    error: str | None
    lifecycle: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LiveStateSnapshot:
    """Strict, correlation-safe view of genuine controller progress."""

    invocation_id: str
    revision: int
    phase: str
    generation: int
    policy_version_used: int
    next_generation: int | None
    next_policy_version: int | None
    worlds_expected: int
    sandboxes_live: int
    experiences_collected: int
    worlds: tuple[LiveWorldSnapshot, ...]
    trainer_metrics: dict[str, Any] | None
    policy_update: dict[str, Any] | None
    generation_json: str | None
    latest_valid: dict[str, Any] | None
    completed_generations: int
    ready_for_next_generation: bool
    interrupted: bool
    error: dict[str, Any] | None
    message: str
    phase_history: tuple[dict[str, Any], ...]
    updated_at: str


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _optional_number(value: Any, label: str) -> float | None:
    return None if value is None else _finite_number(value, label)


def _optional_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    return integer


def _synthetic_sandbox_id(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith(("fixture", "mock", "test", "fake", "local"))


def _valid_action_indices(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(action, int)
            and not isinstance(action, bool)
            and 0 <= action <= 8
            for action in value
        )
    )


def _valid_action_vectors(value: Any, expected_count: int) -> bool:
    if not isinstance(value, list) or len(value) != expected_count:
        return False
    for vector in value:
        if not isinstance(vector, (list, tuple)) or len(vector) != 2:
            return False
        try:
            x = _finite_number(vector[0], "action x")
            y = _finite_number(vector[1], "action y")
        except ValueError:
            return False
        if abs(x) > 1.0 or abs(y) > 1.0:
            return False
    return True


def _controller_daytona_proof(container: dict[str, Any]) -> bool:
    """Recognise the real-only controller envelope without trusting an ID alone."""

    worlds = container.get("worlds")
    if container.get("status") != "COMPLETE" or not isinstance(worlds, list) or not worlds:
        return False
    declared_world_count = container.get("world_count")
    if (
        isinstance(declared_world_count, bool)
        or not isinstance(declared_world_count, int)
        or declared_world_count != len(worlds)
    ):
        return False
    generation = container.get("generation")
    generation_policy = container.get("policy_version")
    next_policy = container.get("next_policy_version")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
        or isinstance(generation_policy, bool)
        or not isinstance(generation_policy, int)
        or generation_policy != generation
        or isinstance(container.get("policy_version_used"), bool)
        or not isinstance(container.get("policy_version_used"), int)
        or container.get("policy_version_used") != generation_policy
        or isinstance(next_policy, bool)
        or not isinstance(next_policy, int)
        or next_policy != generation_policy + 1
    ):
        return False
    generation_extra = container.get("extra")
    required_training_fields = {
        "execution_backend",
        "seed_batch",
        "trainer_checkpoint",
        "training",
        "policy_update",
    }
    if not isinstance(generation_extra, dict) or not required_training_fields.issubset(
        generation_extra
    ):
        return False
    policy_update = generation_extra.get("policy_update")
    if (
        generation_extra.get("execution_backend") != "daytona"
        or not isinstance(generation_extra.get("trainer_checkpoint"), str)
        or not generation_extra["trainer_checkpoint"]
        or not isinstance(generation_extra.get("training"), dict)
        or not generation_extra["training"]
        or not isinstance(policy_update, dict)
        or policy_update.get("weights_changed") is not True
    ):
        return False
    input_digest = policy_update.get("input_model_sha256")
    next_digest = policy_update.get("next_model_sha256")
    input_checkpoint = policy_update.get("input_checkpoint")
    next_checkpoint = policy_update.get("next_checkpoint")
    if (
        not isinstance(input_digest, str)
        or not isinstance(next_digest, str)
        or len(input_digest) != 64
        or len(next_digest) != 64
        or input_digest != input_digest.lower()
        or next_digest != next_digest.lower()
        or input_digest == next_digest
        or any(character not in "0123456789abcdef" for character in input_digest)
        or any(character not in "0123456789abcdef" for character in next_digest)
        or not isinstance(input_checkpoint, str)
        or not input_checkpoint
        or not isinstance(next_checkpoint, str)
        or not next_checkpoint
        or Path(input_checkpoint).name != f"policy_v{generation_policy:03d}.pt"
        or Path(next_checkpoint).name != f"policy_v{next_policy:03d}.pt"
        or generation_extra.get("trainer_checkpoint") != next_checkpoint
    ):
        return False

    sandbox_ids: list[str] = []
    seeds: list[int] = []
    rewards: list[float] = []
    world_indices: list[int] = []
    for world in worlds:
        if not isinstance(world, dict):
            return False
        world_extra = world.get("extra")
        sandbox_id = world.get("sandbox_id")
        if (
            not isinstance(sandbox_id, str)
            or not sandbox_id.strip()
            or _synthetic_sandbox_id(sandbox_id)
            or world.get("execution_backend") != "daytona"
            or not isinstance(world_extra, dict)
            or not isinstance(world_extra.get("universe"), dict)
        ):
            return False
        sandbox_ids.append(sandbox_id)
        try:
            seed = _optional_integer(world.get("seed"), "world seed")
            policy = _optional_integer(world.get("policy_version"), "policy version")
        except ValueError:
            return False
        if seed is None or policy is None or policy != generation_policy:
            return False
        seeds.append(seed)
        try:
            world_index = _optional_integer(world.get("world_index"), "world index")
        except ValueError:
            return False
        if (
            not isinstance(world.get("seed"), int)
            or isinstance(world.get("seed"), bool)
            or not isinstance(world.get("policy_version"), int)
            or isinstance(world.get("policy_version"), bool)
            or not isinstance(world.get("world_index"), int)
            or isinstance(world.get("world_index"), bool)
            or world_index is None
        ):
            return False
        world_indices.append(world_index)
        if not isinstance(world.get("trajectory"), list) or len(world["trajectory"]) < 2:
            return False
        actions = world.get("actions")
        extra = world.get("extra")
        if (
            not _valid_action_indices(actions)
            or not isinstance(extra, dict)
            or not _valid_action_vectors(extra.get("action_vectors"), len(actions))
        ):
            return False
        try:
            rewards.append(_finite_number(world.get("reward"), "world reward"))
        except ValueError:
            return False
        success = world.get("success")
        termination = world.get("termination")
        if (
            not isinstance(success, bool)
            or not isinstance(termination, str)
            or success != (termination == "success")
        ):
            return False

        lifecycle = world.get("lifecycle")
        if not isinstance(lifecycle, list):
            return False
        states: list[str] = []
        for event in lifecycle:
            if not isinstance(event, dict):
                return False
            state = event.get("state")
            state_name = state.upper() if isinstance(state, str) else None
            if isinstance(state, str):
                states.append(state_name)
            event_id = event.get("sandbox_id")
            if state_name == "CREATING":
                if event_id is not None:
                    return False
            elif event_id != sandbox_id:
                return False
            event_seed = event.get("seed")
            if event_seed != seed:
                return False
            event_generation = event.get("generation")
            if event_generation != container.get("generation"):
                return False
            event_policy = event.get("policy_version")
            if event_policy != generation_policy:
                return False
            event_world = event.get("world")
            if event_world != world_index:
                return False
        expected_terminal_state = (
            "SUCCESS"
            if termination == "success"
            else "COLLISION"
            if "collision" in termination
            else termination.upper()
        )
        expected_states = [
            "CREATING",
            "LIVE",
            "RUNNING",
            expected_terminal_state,
            "RESULT_COLLECTED",
        ]
        if (
            expected_terminal_state not in TERMINAL_LIFECYCLE_STATES
            or world.get("status") != expected_terminal_state
            or states != expected_states
        ):
            return False

    seed_batch = generation_extra.get("seed_batch")
    best_index = rewards.index(max(rewards))
    best_world = worlds[best_index]
    best_sandbox_id = sandbox_ids[best_index]
    champion = container.get("champion")
    if not isinstance(champion, dict):
        return False
    try:
        champion_reward = _finite_number(champion.get("reward"), "champion reward")
    except ValueError:
        return False
    champion_matches = (
        isinstance(champion.get("seed"), int)
        and not isinstance(champion.get("seed"), bool)
        and isinstance(champion.get("world_index"), int)
        and not isinstance(champion.get("world_index"), bool)
        and isinstance(champion.get("policy_version"), int)
        and not isinstance(champion.get("policy_version"), bool)
        and isinstance(champion.get("generation"), int)
        and not isinstance(champion.get("generation"), bool)
        and champion.get("sandbox_id") == best_sandbox_id
        and champion.get("seed") == seeds[best_index]
        and champion.get("world_index") == world_indices[best_index]
        and champion.get("policy_version") == generation_policy
        and champion.get("generation") == generation
        and math.isclose(
            champion_reward,
            rewards[best_index],
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
        and champion.get("trajectory") == best_world.get("trajectory")
        and champion.get("actions") == best_world.get("actions")
        and champion.get("success") is best_world.get("success")
        and champion.get("termination") == best_world.get("termination")
        and champion.get("execution_backend") == "daytona"
    )
    try:
        summary_matches = (
            math.isclose(
                _finite_number(container.get("average_reward"), "average reward"),
                math.fsum(rewards) / len(rewards),
                rel_tol=1.0e-12,
                abs_tol=1.0e-9,
            )
            and math.isclose(
                _finite_number(container.get("success_rate"), "success rate"),
                sum(bool(world.get("success")) for world in worlds) / len(worlds),
                rel_tol=1.0e-12,
                abs_tol=1.0e-9,
            )
            and math.isclose(
                _finite_number(container.get("collision_rate"), "collision rate"),
                sum("collision" in str(world.get("termination", "")) for world in worlds)
                / len(worlds),
                rel_tol=1.0e-12,
                abs_tol=1.0e-9,
            )
        )
    except ValueError:
        return False
    return (
        len(sandbox_ids) == len(set(sandbox_ids))
        and len(seeds) == len(set(seeds))
        and world_indices == list(range(1, len(worlds) + 1))
        and isinstance(seed_batch, list)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seed_batch)
        and seed_batch == seeds
        and container.get("best_sandbox_id") == best_sandbox_id
        and isinstance(container.get("best_world"), int)
        and not isinstance(container.get("best_world"), bool)
        and container.get("best_world") == world_indices[best_index]
        and math.isclose(
            _finite_number(container.get("best_reward"), "best reward"),
            max(rewards),
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
        and sandbox_ids[rewards.index(max(rewards))] == best_sandbox_id
        and champion_matches
        and summary_matches
    )


def _raw_daytona_proof(container: dict[str, Any]) -> bool:
    """Validate the direct ``daytona_orchestrator --output`` envelope."""

    summary = container.get("summary")
    results = container.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list) or not results:
        return False
    summary_worlds = summary.get("worlds")
    if (
        isinstance(summary_worlds, bool)
        or not isinstance(summary_worlds, int)
        or summary_worlds != len(results)
    ):
        return False
    ids: list[str] = []
    seeds: list[int] = []
    policies: set[int] = set()
    rewards: list[float] = []
    successes = 0
    for result in results:
        if not isinstance(result, dict):
            return False
        sandbox_id = result.get("sandbox_id")
        if (
            not isinstance(sandbox_id, str)
            or not sandbox_id.strip()
            or _synthetic_sandbox_id(sandbox_id)
        ):
            return False
        ids.append(sandbox_id)
        if not isinstance(result.get("trajectory"), list) or len(result["trajectory"]) < 2:
            return False
        if not isinstance(result.get("universe"), dict):
            return False
        actions = result.get("actions")
        action_vectors = result.get("action_vectors")
        if (
            not _valid_action_indices(actions)
            or not _valid_action_vectors(action_vectors, len(actions))
        ):
            return False
        try:
            rewards.append(_finite_number(result.get("reward"), "world reward"))
            seed = _optional_integer(result.get("seed"), "world seed")
            policy = _optional_integer(result.get("policy_version"), "policy version")
        except ValueError:
            return False
        success = result.get("success")
        termination = result.get("termination")
        if (
            seed is None
            or policy is None
            or not isinstance(result.get("seed"), int)
            or isinstance(result.get("seed"), bool)
            or not isinstance(result.get("policy_version"), int)
            or isinstance(result.get("policy_version"), bool)
            or not isinstance(success, bool)
            or not isinstance(termination, str)
            or success != (termination == "success")
        ):
            return False
        seeds.append(seed)
        policies.add(policy)
        successes += int(success)
    best_sandbox = summary.get("best_sandbox")
    try:
        average_matches = math.isclose(
            _finite_number(summary.get("average_reward"), "average reward"),
            math.fsum(rewards) / len(rewards),
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
        wall_clock_seconds = _finite_number(
            summary.get("wall_clock_seconds"), "wall clock seconds"
        )
    except ValueError:
        return False
    return (
        len(ids) == len(set(ids))
        and len(seeds) == len(set(seeds))
        and len(policies) == 1
        and isinstance(summary.get("successful"), int)
        and not isinstance(summary.get("successful"), bool)
        and summary.get("successful") == successes
        and isinstance(summary.get("seeds"), list)
        and all(
            isinstance(seed, int) and not isinstance(seed, bool)
            for seed in summary["seeds"]
        )
        and summary.get("seeds") == seeds
        and summary.get("sandbox_ids") == ids
        and summary.get("total_trajectory_points")
        == sum(len(result["trajectory"]) for result in results)
        and summary.get("concurrent") is (len(results) > 1)
        and summary.get("cleanup")
        in {"explicit_delete_confirmed", "retained_by_request"}
        and wall_clock_seconds >= 0.0
        and best_sandbox in ids
        and math.isclose(
            _finite_number(summary.get("best_reward"), "best reward"),
            max(rewards),
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
        and ids[rewards.index(max(rewards))] == best_sandbox
        and average_matches
    )


def _rollout_records(payload: Any) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Flatten supported artifacts while retaining their batch-level evidence."""

    if isinstance(payload, list):
        return [(record, {}) for record in payload]
    if not isinstance(payload, dict):
        raise ValueError("rollout JSON must be an object or list")

    recent = payload.get("recent_generations")
    if isinstance(recent, list):
        flattened: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for index, generation in enumerate(recent):
            if not isinstance(generation, dict):
                raise ValueError(f"recent generation {index} must be a JSON object")
            flattened.extend(_rollout_records(generation))
        return flattened

    if "worlds" in payload:
        records = payload["worlds"]
        envelope_kind = "controller"
        verified = _controller_daytona_proof(payload)
    elif "results" in payload:
        records = payload["results"]
        envelope_kind = "raw_daytona"
        verified = _raw_daytona_proof(payload)
    elif "rollouts" in payload:
        records = payload["rollouts"]
        envelope_kind = "recorded"
        verified = False
    elif "trajectory" in payload:
        records = [payload]
        envelope_kind = "recorded"
        verified = False
    else:
        raise ValueError(
            "rollout JSON needs 'worlds', 'results', 'rollouts', or one trajectory"
        )
    if not isinstance(records, list):
        key = next(key for key in ("worlds", "results", "rollouts") if key in payload)
        raise ValueError(f"'{key}' must be a list")

    champion = payload.get("champion")
    champion_sandbox = payload.get("best_sandbox_id")
    champion_seed = None
    champion_world = payload.get("best_world")
    if isinstance(champion, dict):
        champion_sandbox = champion.get("sandbox_id", champion_sandbox)
        champion_seed = champion.get("seed")
        champion_world = champion.get("world_index", champion_world)
    if envelope_kind == "raw_daytona" and isinstance(payload.get("summary"), dict):
        champion_sandbox = payload["summary"].get("best_sandbox", champion_sandbox)

    metric_source = (
        payload.get("summary", {})
        if envelope_kind == "raw_daytona"
        else payload
    )
    if not isinstance(metric_source, dict):
        metric_source = {}
    success_rate = metric_source.get("success_rate")
    successful = metric_source.get("successful")
    metric_worlds = (
        metric_source.get("worlds")
        if envelope_kind == "raw_daytona"
        else payload.get("world_count")
    )
    if (
        success_rate is None
        and isinstance(successful, int)
        and isinstance(metric_worlds, int)
        and metric_worlds > 0
    ):
        success_rate = successful / metric_worlds

    context = {
        "generation": payload.get("generation"),
        "policy_version": payload.get("policy_version"),
        "next_policy_version": payload.get("next_policy_version"),
        "champion_sandbox": champion_sandbox,
        "champion_seed": champion_seed,
        "champion_world": champion_world,
        "envelope_kind": envelope_kind,
        "verified": verified,
        "average_reward": metric_source.get("average_reward"),
        "best_reward": metric_source.get("best_reward"),
        "success_rate": success_rate,
        "collision_rate": metric_source.get("collision_rate"),
        "world_count": metric_worlds,
    }
    return [(record, context) for record in records]


def load_rollout_trails(path: str | Path) -> list[AttemptTrail]:
    """Load recorded worker output without inventing values or provenance."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = _rollout_records(payload)
    if not records:
        raise ValueError("rollout artifact contains no recorded trajectories")

    attempts: list[AttemptTrail] = []
    for index, (rollout, context) in enumerate(records):
        if not isinstance(rollout, dict):
            raise ValueError(f"rollout {index} must be a JSON object")
        trajectory = rollout.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) < 2:
            raise ValueError(f"rollout {index} needs a trajectory with at least two points")

        points: list[tuple[float, float]] = []
        normalized_trajectory: list[dict[str, Any]] = []
        for point_index, point in enumerate(trajectory):
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                raise ValueError(f"rollout {index} trajectory point {point_index} needs x and y")
            normalized = dict(point)
            normalized["x"] = _finite_number(point["x"], f"rollout {index} point {point_index} x")
            normalized["y"] = _finite_number(point["y"], f"rollout {index} point {point_index} y")
            for field in ("vx", "vy", "reward", "clearance"):
                if field in point and point[field] is not None:
                    normalized[field] = _finite_number(
                        point[field], f"rollout {index} point {point_index} {field}"
                    )
            if "step" in point and point["step"] is not None:
                normalized["step"] = _optional_integer(
                    point["step"], f"rollout {index} point {point_index} step"
                )
            points.append((normalized["x"], normalized["y"]))
            normalized_trajectory.append(normalized)

        sandbox_id = rollout.get("sandbox_id")
        if sandbox_id is not None and (
            not isinstance(sandbox_id, str) or not sandbox_id.strip()
        ):
            raise ValueError(f"rollout {index} sandbox_id must be a non-empty string")
        reward = rollout.get("reward")
        success = rollout.get("success")
        if success is not None and not isinstance(success, bool):
            raise ValueError(f"rollout {index} success must be a JSON boolean")
        termination = rollout.get("termination")
        if termination is not None and (
            not isinstance(termination, str) or not termination.strip()
        ):
            raise ValueError(f"rollout {index} termination must be non-empty text")
        if (
            success is not None
            and termination is not None
            and success != (termination.lower() == "success")
        ):
            raise ValueError(
                f"rollout {index} success and termination describe different outcomes"
            )
        rollout_seed = rollout.get("seed")
        rollout_policy = rollout.get("policy_version", context.get("policy_version"))
        next_policy = rollout.get(
            "next_policy_version", context.get("next_policy_version")
        )
        rollout_generation = rollout.get("generation", context.get("generation"))
        extra = rollout.get("extra") if isinstance(rollout.get("extra"), dict) else {}
        raw_universe = rollout.get("universe", extra.get("universe"))
        if raw_universe is not None and not isinstance(raw_universe, dict):
            raise ValueError(f"rollout {index} universe must be a JSON object")
        raw_lifecycle = rollout.get("lifecycle", [])
        if raw_lifecycle is None:
            raw_lifecycle = []
        if not isinstance(raw_lifecycle, list):
            raise ValueError(f"rollout {index} lifecycle must be a list")
        lifecycle: list[dict[str, Any]] = []
        for event_index, event in enumerate(raw_lifecycle):
            if not isinstance(event, dict):
                raise ValueError(
                    f"rollout {index} lifecycle event {event_index} must be an object"
                )
            state = event.get("state")
            if state is not None and not isinstance(state, str):
                raise ValueError(
                    f"rollout {index} lifecycle event {event_index} state must be text"
                )
            lifecycle.append(dict(event))
        raw_actions = rollout.get("action_vectors", extra.get("action_vectors", []))
        if raw_actions is None:
            raw_actions = []
        if not isinstance(raw_actions, list):
            raise ValueError(f"rollout {index} action_vectors must be a list")
        action_vectors: list[tuple[float, float]] = []
        for action_index, action in enumerate(raw_actions):
            if not isinstance(action, (list, tuple)) or len(action) != 2:
                raise ValueError(f"rollout {index} action vector {action_index} needs x and y")
            action_x = _finite_number(action[0], f"rollout {index} action {action_index} x")
            action_y = _finite_number(action[1], f"rollout {index} action {action_index} y")
            if abs(action_x) > 1.0 or abs(action_y) > 1.0:
                raise ValueError(f"rollout {index} action vector {action_index} is outside [-1, 1]")
            action_vectors.append((action_x, action_y))

        point_speeds = [
            math.hypot(float(point["vx"]), float(point["vy"]))
            for point in normalized_trajectory
            if point.get("vx") is not None and point.get("vy") is not None
        ]
        point_clearances = [
            float(point["clearance"])
            for point in normalized_trajectory
            if point.get("clearance") is not None
        ]
        top_min_clearance = rollout.get("min_clearance")
        top_mean_speed = rollout.get("mean_speed")
        top_max_speed = rollout.get("max_speed")
        world_index = _optional_integer(rollout.get("world_index"), f"rollout {index} world_index")
        rollout_seed_value = _optional_integer(rollout_seed, f"rollout {index} seed")
        rollout_policy_value = _optional_integer(
            rollout_policy, f"rollout {index} policy_version"
        )
        next_policy_value = _optional_integer(
            next_policy, f"rollout {index} next_policy_version"
        )
        rollout_generation_value = _optional_integer(
            rollout_generation, f"rollout {index} generation"
        )
        generation_world_count = _optional_integer(
            context.get("world_count"), f"rollout {index} generation world_count"
        )
        if generation_world_count is not None and generation_world_count <= 0:
            raise ValueError(f"rollout {index} generation world_count must be positive")
        generation_average_reward = _optional_number(
            context.get("average_reward"),
            f"rollout {index} generation average_reward",
        )
        generation_best_reward = _optional_number(
            context.get("best_reward"),
            f"rollout {index} generation best_reward",
        )
        generation_success_rate = _optional_number(
            context.get("success_rate"),
            f"rollout {index} generation success_rate",
        )
        generation_collision_rate = _optional_number(
            context.get("collision_rate"),
            f"rollout {index} generation collision_rate",
        )
        for rate_name, rate in (
            ("success_rate", generation_success_rate),
            ("collision_rate", generation_collision_rate),
        ):
            if rate is not None and not 0.0 <= rate <= 1.0:
                raise ValueError(
                    f"rollout {index} generation {rate_name} must be between 0 and 1"
                )
        explicit_champion = (
            sandbox_id is not None
            and sandbox_id == context.get("champion_sandbox")
        ) or (
            context.get("champion_sandbox") is None
            and context.get("champion_world") is not None
            and world_index == context.get("champion_world")
        ) or (
            context.get("champion_sandbox") is None
            and context.get("champion_world") is None
            and context.get("champion_seed") is not None
            and rollout_seed_value == context.get("champion_seed")
        )
        champion_declared = any(
            context.get(field) is not None
            for field in ("champion_sandbox", "champion_world", "champion_seed")
        )
        verified = bool(context.get("verified"))
        envelope_kind = context.get("envelope_kind")
        if verified and envelope_kind == "controller":
            provenance = "DAYTONA TRAINING"
        elif verified and envelope_kind == "raw_daytona":
            provenance = "DAYTONA ROLLOUT"
        else:
            # Anything loaded through --rollouts is recorded input. Local
            # preview is reserved for the explicit --local-preview runtime.
            provenance = "UNVERIFIED RECORDED REPLAY"
        attempts.append(
            AttemptTrail(
                points=tuple(points),
                reward=_optional_number(reward, f"rollout {index} reward"),
                success=success,
                seed=rollout_seed_value,
                policy_version=rollout_policy_value,
                next_policy_version=next_policy_value,
                generation=rollout_generation_value,
                sandbox_id=sandbox_id,
                trajectory=tuple(normalized_trajectory),
                action_vectors=tuple(action_vectors),
                universe=(dict(raw_universe) if raw_universe is not None else None),
                lifecycle=tuple(lifecycle),
                world_index=world_index,
                termination=termination,
                min_clearance=(
                    _optional_number(top_min_clearance, f"rollout {index} min_clearance")
                    if top_min_clearance is not None
                    else min(point_clearances, default=None)
                ),
                mean_speed=(
                    _optional_number(top_mean_speed, f"rollout {index} mean_speed")
                    if top_mean_speed is not None
                    else (sum(point_speeds) / len(point_speeds) if point_speeds else None)
                ),
                max_speed=(
                    _optional_number(top_max_speed, f"rollout {index} max_speed")
                    if top_max_speed is not None
                    else max(point_speeds, default=None)
                ),
                generation_average_reward=generation_average_reward,
                generation_best_reward=generation_best_reward,
                generation_success_rate=generation_success_rate,
                generation_collision_rate=generation_collision_rate,
                generation_world_count=generation_world_count,
                provenance=provenance,
                daytona_verified=verified,
                is_champion=bool(explicit_champion),
                champion_declared=champion_declared,
                source_file=str(source.resolve()),
            )
        )

    # Compact/local artifacts may not declare a champion. In that case only,
    # use their recorded rewards within each generation/policy group.
    group_indices: dict[tuple[int | None, int | None], list[int]] = {}
    for index, attempt in enumerate(attempts):
        group_indices.setdefault((attempt.generation, attempt.policy_version), []).append(index)
    for indices in group_indices.values():
        declared = [index for index in indices if attempts[index].champion_declared]
        explicit = [index for index in indices if attempts[index].is_champion]
        if len(explicit) > 1:
            raise ValueError("rollout artifact declares multiple champions in one generation")
        if declared and len(explicit) != 1:
            raise ValueError("rollout artifact champion does not identify exactly one world")
        if explicit:
            champion_reward = attempts[explicit[0]].reward
            scored_rewards = [
                float(attempts[index].reward)
                for index in indices
                if attempts[index].reward is not None
            ]
            if (
                champion_reward is None
                or not scored_rewards
                or not math.isclose(
                    float(champion_reward),
                    max(scored_rewards),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-9,
                )
            ):
                raise ValueError(
                    "rollout artifact champion is not a maximum recorded reward"
                )
            continue
        scored = [index for index in indices if attempts[index].reward is not None]
        if scored:
            champion_index = max(scored, key=lambda item: float(attempts[item].reward))
            attempts[champion_index] = replace(attempts[champion_index], is_champion=True)
    return attempts


def _json_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} JSON integer")
    return value


def _json_boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _optional_live_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text when present")
    return value


def _optional_live_integer(value: Any, label: str) -> int | None:
    return None if value is None else _json_integer(value, label)


def _live_lifecycle_sequence(states: Sequence[str], label: str) -> None:
    """Require a real lifecycle prefix, with ERROR allowed only as the last event."""

    remaining = list(states)
    if remaining and remaining[-1] == "ERROR":
        remaining.pop()
    elif "ERROR" in remaining:
        raise ValueError(f"{label} ERROR must be the final lifecycle event")
    if not remaining:
        return
    if len(remaining) > 5:
        raise ValueError(f"{label} contains too many lifecycle events")
    if remaining[0] != "CREATING":
        raise ValueError(f"{label} must begin with CREATING")
    if len(remaining) >= 2 and remaining[1] != "LIVE":
        raise ValueError(f"{label} must progress from CREATING to LIVE")
    if len(remaining) >= 3 and remaining[2] != "RUNNING":
        raise ValueError(f"{label} must progress from LIVE to RUNNING")
    if len(remaining) >= 4 and remaining[3] not in TERMINAL_LIFECYCLE_STATES:
        raise ValueError(f"{label} needs a real terminal outcome after RUNNING")
    if len(remaining) >= 5 and remaining[4] != "RESULT_COLLECTED":
        raise ValueError(f"{label} must finish with RESULT_COLLECTED")


def _validate_live_world(
    raw: Any,
    *,
    generation: int,
    policy_version: int,
    expected_index: int,
) -> LiveWorldSnapshot:
    if not isinstance(raw, Mapping):
        raise ValueError(f"live world {expected_index} must be a JSON object")
    world_index = _json_integer(
        raw.get("world_index"), f"live world {expected_index} world_index", minimum=1
    )
    if world_index != expected_index:
        raise ValueError(
            f"live world index {world_index} does not match slot {expected_index}"
        )
    seed = _optional_live_integer(raw.get("seed"), f"live world {world_index} seed")
    sandbox_id = _optional_live_text(
        raw.get("sandbox_id"), f"live world {world_index} sandbox_id"
    )
    if sandbox_id is not None and _synthetic_sandbox_id(sandbox_id):
        raise ValueError(f"live world {world_index} has a synthetic sandbox ID")
    lifecycle_state = _optional_live_text(
        raw.get("lifecycle_state"), f"live world {world_index} lifecycle_state"
    )
    if lifecycle_state is not None:
        lifecycle_state = lifecycle_state.upper()
        if lifecycle_state not in LIVE_WORLD_STATES:
            raise ValueError(
                f"live world {world_index} has unknown state {lifecycle_state}"
            )
    reward = _optional_number(raw.get("reward"), f"live world {world_index} reward")
    success = raw.get("success")
    if success is not None and not isinstance(success, bool):
        raise ValueError(f"live world {world_index} success must be a JSON boolean")
    termination = _optional_live_text(
        raw.get("termination"), f"live world {world_index} termination"
    )
    result_collected = _json_boolean(
        raw.get("result_collected"), f"live world {world_index} result_collected"
    )
    error = _optional_live_text(raw.get("error"), f"live world {world_index} error")
    raw_lifecycle = raw.get("lifecycle")
    if not isinstance(raw_lifecycle, list):
        raise ValueError(f"live world {world_index} lifecycle must be a list")

    lifecycle: list[dict[str, Any]] = []
    states: list[str] = []
    event_sandbox_id: str | None = None
    event_seed: int | None = None
    event_reward: float | None = None
    for event_index, event in enumerate(raw_lifecycle):
        if not isinstance(event, Mapping):
            raise ValueError(
                f"live world {world_index} lifecycle event {event_index} must be an object"
            )
        event_generation = _json_integer(
            event.get("generation"),
            f"live world {world_index} lifecycle generation",
        )
        event_policy = _json_integer(
            event.get("policy_version"),
            f"live world {world_index} lifecycle policy_version",
        )
        event_world = _json_integer(
            event.get("world"), f"live world {world_index} lifecycle world", minimum=1
        )
        if (
            event_generation != generation
            or event_policy != policy_version
            or event_world != world_index
        ):
            raise ValueError(
                f"live world {world_index} lifecycle identity does not match the active run"
            )
        current_seed = _json_integer(
            event.get("seed"), f"live world {world_index} lifecycle seed"
        )
        if event_seed is not None and current_seed != event_seed:
            raise ValueError(f"live world {world_index} lifecycle seed changed")
        event_seed = current_seed
        state = _optional_live_text(
            event.get("state"), f"live world {world_index} lifecycle state"
        )
        assert state is not None
        state = state.upper()
        if state not in LIVE_WORLD_STATES:
            raise ValueError(f"live world {world_index} has unknown lifecycle state {state}")
        states.append(state)
        current_sandbox = _optional_live_text(
            event.get("sandbox_id"),
            f"live world {world_index} lifecycle sandbox_id",
        )
        if current_sandbox is not None and _synthetic_sandbox_id(current_sandbox):
            raise ValueError(f"live world {world_index} lifecycle has a synthetic ID")
        if state == "CREATING" and current_sandbox is not None:
            raise ValueError(f"live world {world_index} CREATING cannot claim an ID")
        if state not in {"CREATING", "ERROR"} and current_sandbox is None:
            raise ValueError(f"live world {world_index} {state} requires its real ID")
        if current_sandbox is not None:
            if event_sandbox_id is not None and current_sandbox != event_sandbox_id:
                raise ValueError(f"live world {world_index} lifecycle sandbox ID changed")
            event_sandbox_id = current_sandbox
        if event.get("reward") is not None:
            event_reward = _finite_number(
                event["reward"], f"live world {world_index} lifecycle reward"
            )
        lifecycle.append(dict(event))

    _live_lifecycle_sequence(states, f"live world {world_index} lifecycle")
    if lifecycle:
        if lifecycle_state != states[-1]:
            raise ValueError(
                f"live world {world_index} current state does not match its lifecycle"
            )
        if seed != event_seed:
            raise ValueError(f"live world {world_index} seed does not match its lifecycle")
        if sandbox_id != event_sandbox_id:
            raise ValueError(
                f"live world {world_index} sandbox ID does not match its lifecycle"
            )
    elif lifecycle_state is not None or seed is not None or sandbox_id is not None:
        raise ValueError(
            f"live world {world_index} cannot claim progress without lifecycle evidence"
        )
    collected_in_lifecycle = "RESULT_COLLECTED" in states
    if result_collected != collected_in_lifecycle:
        raise ValueError(
            f"live world {world_index} result_collected disagrees with its lifecycle"
        )
    if reward is not None and event_reward is not None and not math.isclose(
        reward, event_reward, rel_tol=1.0e-12, abs_tol=1.0e-9
    ):
        raise ValueError(f"live world {world_index} reward changed across lifecycle fields")
    if success is not None and termination is not None:
        if success != (termination.lower() == "success"):
            raise ValueError(
                f"live world {world_index} success and termination disagree"
            )
    if lifecycle_state == "ERROR" and error is None:
        raise ValueError(f"live world {world_index} ERROR requires a real error message")
    return LiveWorldSnapshot(
        world_index=world_index,
        seed=seed,
        sandbox_id=sandbox_id,
        lifecycle_state=lifecycle_state,
        reward=reward,
        success=success,
        termination=termination,
        result_collected=result_collected,
        error=error,
        lifecycle=tuple(lifecycle),
    )


def _validate_live_metrics(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("live trainer_metrics must be a JSON object")
    metrics: dict[str, Any] = {}
    integer_metrics = {
        "episodes",
        "transitions",
        "changed_parameter_tensors",
        "changed_parameter_elements",
        "changed_tensors",
        "changed_elements",
    }
    for key in LIVE_TRAINER_METRICS:
        if key not in raw:
            continue
        if key == "weights_changed":
            metrics[key] = _json_boolean(raw[key], f"trainer_metrics.{key}")
        elif key in integer_metrics:
            metrics[key] = _json_integer(raw[key], f"trainer_metrics.{key}")
        else:
            metrics[key] = _finite_number(raw[key], f"trainer_metrics.{key}")
    for alias, canonical in (
        ("changed_tensors", "changed_parameter_tensors"),
        ("changed_elements", "changed_parameter_elements"),
    ):
        if alias in metrics and canonical in metrics and metrics[alias] != metrics[canonical]:
            raise ValueError(f"trainer_metrics.{alias} disagrees with {canonical}")
    return metrics


def _validate_live_policy_update(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("live policy_update must be a JSON object")
    required = {
        "input_checkpoint",
        "input_model_sha256",
        "next_checkpoint",
        "next_model_sha256",
        "weights_changed",
    }
    if not required.issubset(raw):
        missing = ", ".join(sorted(required.difference(raw)))
        raise ValueError(f"live policy_update is missing {missing}")
    update = {key: raw[key] for key in required}
    for key in ("input_checkpoint", "next_checkpoint"):
        _optional_live_text(update[key], f"policy_update.{key}")
    for key in ("input_model_sha256", "next_model_sha256"):
        digest = _optional_live_text(update[key], f"policy_update.{key}")
        assert digest is not None
        if (
            len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"policy_update.{key} must be a lowercase SHA-256 digest")
    if update["input_model_sha256"] == update["next_model_sha256"]:
        raise ValueError("live policy update has identical model-state digests")
    _json_boolean(update["weights_changed"], "policy_update.weights_changed")
    return update


def _validate_live_state(
    payload: Any,
    *,
    expected_invocation_id: str,
    expected_generation: int,
    expected_policy_version: int,
    expected_worlds: int,
    expected_max_steps: int,
    last_revision: int = -1,
) -> LiveStateSnapshot | None:
    """Validate one atomic controller snapshot; stale invocations are ignored."""

    if not isinstance(payload, Mapping):
        raise ValueError("live state must be a JSON object")
    invocation_id = payload.get("invocation_id")
    if not isinstance(invocation_id, str) or not invocation_id:
        raise ValueError("live state invocation_id must be non-empty text")
    if invocation_id != expected_invocation_id:
        return None
    revision = _json_integer(payload.get("revision"), "live state revision", minimum=1)
    if revision <= last_revision:
        return None
    if payload.get("schema_version") != LIVE_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported live-state schema_version")
    if payload.get("mode") != "live_daytona_training":
        raise ValueError("live state mode is not live_daytona_training")
    if payload.get("execution_backend") != "daytona":
        raise ValueError("live state does not identify the Daytona backend")
    if _json_integer(payload.get("requested_generations"), "requested_generations", minimum=1) != 1:
        raise ValueError("one R press must request exactly one generation")
    if _json_integer(payload.get("requested_worlds"), "requested_worlds", minimum=1) != expected_worlds:
        raise ValueError("live state requested_worlds does not match the UI launch")
    if _json_integer(payload.get("max_steps"), "max_steps", minimum=1) != expected_max_steps:
        raise ValueError("live state max_steps does not match the UI launch")
    generation = _json_integer(payload.get("generation"), "live state generation")
    policy_version = _json_integer(
        payload.get("policy_version_used"), "live state policy_version_used"
    )
    if generation != expected_generation or policy_version != expected_policy_version:
        raise ValueError("live state generation/policy does not match the frozen launch")
    if generation != policy_version:
        raise ValueError("canonical live generation must equal its frozen policy version")
    phase = _optional_live_text(payload.get("phase"), "live state phase")
    assert phase is not None
    phase = phase.upper()
    if phase not in LIVE_TRAINING_PHASES:
        raise ValueError(f"unknown live-training phase {phase}")
    worlds_expected = _json_integer(
        payload.get("worlds_expected"), "worlds_expected", minimum=1
    )
    if worlds_expected != expected_worlds:
        raise ValueError("live state worlds_expected does not match the launch")
    sandboxes_live = _json_integer(payload.get("sandboxes_live"), "sandboxes_live")
    experiences_collected = _json_integer(
        payload.get("experiences_collected"), "experiences_collected"
    )
    if sandboxes_live > worlds_expected or experiences_collected > worlds_expected:
        raise ValueError("live world progress exceeds the requested world count")

    raw_worlds = payload.get("worlds")
    if not isinstance(raw_worlds, list):
        raise ValueError("live state worlds must be a list")
    if phase == "READY" and not raw_worlds:
        worlds: tuple[LiveWorldSnapshot, ...] = ()
    else:
        if len(raw_worlds) != worlds_expected:
            raise ValueError("live state must retain one slot per requested world")
        worlds = tuple(
            _validate_live_world(
                raw,
                generation=generation,
                policy_version=policy_version,
                expected_index=index,
            )
            for index, raw in enumerate(raw_worlds, start=1)
        )
        seeds = [world.seed for world in worlds if world.seed is not None]
        ids = [world.sandbox_id for world in worlds if world.sandbox_id is not None]
        if len(seeds) != len(set(seeds)):
            raise ValueError("live worlds contain duplicate universe seeds")
        if len(ids) != len(set(ids)):
            raise ValueError("live worlds contain duplicate sandbox IDs")
        if sandboxes_live != len(ids):
            raise ValueError("sandboxes_live does not match real sandbox identities")
        if experiences_collected != sum(world.result_collected for world in worlds):
            raise ValueError("experiences_collected does not match world lifecycle evidence")

    next_generation = _optional_live_integer(
        payload.get("next_generation"), "next_generation"
    )
    next_policy_version = _optional_live_integer(
        payload.get("next_policy_version"), "next_policy_version"
    )
    trainer_metrics = _validate_live_metrics(payload.get("trainer_metrics"))
    policy_update = _validate_live_policy_update(payload.get("policy_update"))
    generation_json = _optional_live_text(
        payload.get("generation_json"), "generation_json"
    )
    latest_valid_raw = payload.get("latest_valid")
    if latest_valid_raw is not None and not isinstance(latest_valid_raw, Mapping):
        raise ValueError("latest_valid must be a JSON object when present")
    latest_valid = None if latest_valid_raw is None else dict(latest_valid_raw)
    completed_generations = _json_integer(
        payload.get("completed_generations"), "completed_generations"
    )
    ready_for_next = _json_boolean(
        payload.get("ready_for_next_generation"), "ready_for_next_generation"
    )
    interrupted = _json_boolean(payload.get("interrupted"), "interrupted")
    error_raw = payload.get("error")
    if error_raw is not None and not isinstance(error_raw, Mapping):
        raise ValueError("live state error must be a JSON object when present")
    error = None if error_raw is None else dict(error_raw)
    message = _optional_live_text(payload.get("message"), "live state message")
    assert message is not None
    updated_at = _optional_live_text(payload.get("updated_at"), "live state updated_at")
    assert updated_at is not None

    raw_history = payload.get("phase_history")
    if not isinstance(raw_history, list) or not raw_history:
        raise ValueError("live state phase_history must retain real controller transitions")
    history: list[dict[str, Any]] = []
    previous_history_revision = 0
    for index, item in enumerate(raw_history):
        if not isinstance(item, Mapping):
            raise ValueError(f"phase_history[{index}] must be a JSON object")
        history_revision = _json_integer(
            item.get("revision"), f"phase_history[{index}].revision", minimum=1
        )
        history_phase = _optional_live_text(
            item.get("phase"), f"phase_history[{index}].phase"
        )
        assert history_phase is not None
        history_phase = history_phase.upper()
        if history_phase not in LIVE_TRAINING_PHASES:
            raise ValueError(f"phase_history[{index}] has an unknown phase")
        if history_revision <= previous_history_revision or history_revision > revision:
            raise ValueError("phase_history revisions must be increasing and current")
        if _json_integer(item.get("generation"), f"phase_history[{index}].generation") != generation:
            raise ValueError("phase_history generation changed within one invocation")
        if _json_integer(
            item.get("policy_version_used"),
            f"phase_history[{index}].policy_version_used",
        ) != policy_version:
            raise ValueError("phase_history policy changed within one frozen generation")
        previous_history_revision = history_revision
        normalized_item = dict(item)
        normalized_item["phase"] = history_phase
        history.append(normalized_item)

    if phase in {"POLICY_UPDATED", "GENERATION_COMPLETE"}:
        if (
            next_generation != generation + 1
            or next_policy_version != policy_version + 1
        ):
            raise ValueError("completed live policy must advance exactly one version")
        if experiences_collected != worlds_expected:
            raise ValueError("policy update cannot precede all collected experiences")
        if trainer_metrics is None or trainer_metrics.get("weights_changed") is not True:
            raise ValueError("live trainer did not prove changed weights")
        if policy_update is None or policy_update.get("weights_changed") is not True:
            raise ValueError("live checkpoint proof did not prove changed weights")
        if generation_json is None or latest_valid is None:
            raise ValueError("updated live state must name its exact persisted generation")
    if phase == "GENERATION_COMPLETE":
        if completed_generations != 1 or not ready_for_next:
            raise ValueError("terminal live generation is not marked ready for the next R press")
    elif phase != "READY" and ready_for_next:
        raise ValueError("live state unlocked R before terminal success")
    if phase == "ERROR":
        if error is None or ready_for_next:
            raise ValueError("failed live state must carry an error and remain unready")
    elif error is not None:
        raise ValueError("non-error live state cannot carry a controller error")

    return LiveStateSnapshot(
        invocation_id=invocation_id,
        revision=revision,
        phase=phase,
        generation=generation,
        policy_version_used=policy_version,
        next_generation=next_generation,
        next_policy_version=next_policy_version,
        worlds_expected=worlds_expected,
        sandboxes_live=sandboxes_live,
        experiences_collected=experiences_collected,
        worlds=worlds,
        trainer_metrics=trainer_metrics,
        policy_update=policy_update,
        generation_json=generation_json,
        latest_valid=latest_valid,
        completed_generations=completed_generations,
        ready_for_next_generation=ready_for_next,
        interrupted=interrupted,
        error=error,
        message=message,
        phase_history=tuple(history),
        updated_at=updated_at,
    )


def _read_live_state(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_live_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else repo_root / candidate).resolve()


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _load_completed_live_generation(
    state: LiveStateSnapshot,
    config: LiveTrainingConfig,
) -> tuple[list[AttemptTrail], Path]:
    """Strictly load the exact terminal artifact before any visible data swap."""

    if state.phase != "GENERATION_COMPLETE" or not state.ready_for_next_generation:
        raise ValueError("live generation is not terminal and ready")
    assert state.generation_json is not None
    generation_path = _resolve_live_path(state.generation_json, repo_root=config.repo_root)
    runs_dir = config.runs_dir.resolve()
    if not _path_is_within(generation_path, runs_dir):
        raise ValueError("completed generation JSON is outside the configured runs directory")
    expected_name = f"generation_{state.generation:03d}.json"
    if generation_path.name != expected_name:
        raise ValueError(
            f"completed generation path must identify {expected_name}, not {generation_path.name}"
        )

    attempts = load_rollout_trails(generation_path)
    if len(attempts) != state.worlds_expected:
        raise ValueError("completed generation world count differs from live state")
    if not attempts or not all(attempt.daytona_verified for attempt in attempts):
        raise ValueError("completed live artifact failed strict Daytona provenance checks")
    if any(
        attempt.generation != state.generation
        or attempt.policy_version != state.policy_version_used
        or attempt.next_policy_version != state.next_policy_version
        for attempt in attempts
    ):
        raise ValueError("completed artifact identity differs from the live frozen policy")
    _validated_replay_environments(attempts)

    with generation_path.open("r", encoding="utf-8") as handle:
        generation_payload = json.load(handle)
    extra = generation_payload.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError("completed generation is missing its training proof")
    artifact_metrics = extra.get("training")
    artifact_update = extra.get("policy_update")
    if not isinstance(artifact_metrics, Mapping) or not isinstance(artifact_update, Mapping):
        raise ValueError("completed generation is missing trainer/checkpoint proof")
    assert state.trainer_metrics is not None
    assert state.policy_update is not None
    for key, value in state.trainer_metrics.items():
        canonical = {
            "changed_tensors": "changed_parameter_tensors",
            "changed_elements": "changed_parameter_elements",
        }.get(key, key)
        if artifact_metrics.get(canonical) != value:
            raise ValueError(f"live trainer metric {key} differs from the generation artifact")
    for key, value in state.policy_update.items():
        if artifact_update.get(key) != value:
            raise ValueError(f"live policy update field {key} differs from the artifact")

    next_policy = state.next_policy_version
    assert next_policy is not None
    checkpoint_value = state.policy_update["next_checkpoint"]
    checkpoint_path = _resolve_live_path(checkpoint_value, repo_root=config.repo_root)
    if not _path_is_within(checkpoint_path, config.checkpoint_dir.resolve()):
        raise ValueError("next policy checkpoint is outside the configured checkpoint directory")
    if checkpoint_path.name != f"policy_v{next_policy:03d}.pt":
        raise ValueError("next checkpoint filename does not match the advanced policy version")
    if not checkpoint_path.is_file():
        raise ValueError("verified next policy checkpoint is missing from disk")
    latest = state.latest_valid
    assert latest is not None
    expected_latest = {
        "generation": state.generation,
        "policy_version_used": state.policy_version_used,
        "next_policy_version": next_policy,
        "generation_json": state.generation_json,
        "checkpoint": checkpoint_value,
        "checkpoint_model_sha256": state.policy_update["next_model_sha256"],
    }
    if any(latest.get(key) != value for key, value in expected_latest.items()):
        raise ValueError("latest_valid does not bind the completed generation/checkpoint")
    return attempts, checkpoint_path


def _live_controller_argv(
    config: LiveTrainingConfig,
    invocation_id: str,
    *,
    generation: int,
    policy_version: int,
    checkpoint: Path | None,
) -> list[str]:
    """Build one shell-free GRAV 3 invocation; the environment is inherited."""

    argv = [
        sys.executable,
        "-m",
        LIVE_CONTROLLER_MODULE,
        "--generations",
        "1",
        "--worlds",
        str(config.worlds),
        "--max-steps",
        str(config.max_steps),
        "--base-seed",
        str(config.base_seed),
        "--runs-dir",
        str(config.runs_dir),
        "--checkpoint-dir",
        str(config.checkpoint_dir),
        "--snapshot-name",
        config.snapshot_name,
        "--live-state-path",
        str(config.live_state_path),
        "--invocation-id",
        invocation_id,
        "--start-generation",
        str(generation),
        "--initial-policy-version",
        str(policy_version),
    ]
    if checkpoint is not None:
        argv.extend(("--policy-checkpoint", str(checkpoint)))
    return argv


@dataclass
class LiveTrainingSession:
    """Nonblocking process/poll state for the judge-facing live window."""

    config: LiveTrainingConfig
    process: Any | None = field(default=None, init=False)
    invocation_id: str | None = field(default=None, init=False)
    state: LiveStateSnapshot | None = field(default=None, init=False)
    last_revision: int = field(default=-1, init=False)
    next_poll_at: float = field(default=0.0, init=False)
    current_generation: int = field(init=False)
    current_policy_version: int = field(init=False)
    current_checkpoint: Path | None = field(init=False)
    error: str | None = field(default=None, init=False)
    warning: str | None = field(default=None, init=False)
    locked_notice: bool = field(default=False, init=False)
    _completed: tuple[list[AttemptTrail], Path] | None = field(default=None, init=False)
    _seen_phase_revision: int = field(default=0, init=False)
    _phase_queue: deque[str] = field(default_factory=deque, init=False)
    _display_phase: str | None = field(default=None, init=False)
    _display_phase_until: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.current_generation = self.config.start_generation
        self.current_policy_version = self.config.policy_version
        self.current_checkpoint = self.config.start_checkpoint

    @property
    def active(self) -> bool:
        return self.process is not None

    def request_generation(
        self,
        *,
        popen_factory: Callable[..., Any] | None = None,
    ) -> bool:
        if self.active:
            self.locked_notice = True
            return False
        invocation_id = uuid.uuid4().hex
        argv = _live_controller_argv(
            self.config,
            invocation_id,
            generation=self.current_generation,
            policy_version=self.current_policy_version,
            checkpoint=self.current_checkpoint,
        )
        launcher = subprocess.Popen if popen_factory is None else popen_factory
        self.process = launcher(
            argv,
            cwd=str(self.config.repo_root),
            shell=False,
        )
        self.invocation_id = invocation_id
        self.state = None
        self.last_revision = -1
        self.next_poll_at = 0.0
        self.error = None
        self.warning = None
        self.locked_notice = False
        self._completed = None
        self._seen_phase_revision = 0
        self._phase_queue.clear()
        self._display_phase = "READY"
        self._display_phase_until = 0.0
        return True

    def _accept_state(self, payload: Any) -> None:
        assert self.invocation_id is not None
        snapshot = _validate_live_state(
            payload,
            expected_invocation_id=self.invocation_id,
            expected_generation=self.current_generation,
            expected_policy_version=self.current_policy_version,
            expected_worlds=self.config.worlds,
            expected_max_steps=self.config.max_steps,
            last_revision=self.last_revision,
        )
        if snapshot is None:
            return
        self.state = snapshot
        self.last_revision = snapshot.revision
        self.warning = None
        for item in snapshot.phase_history:
            item_revision = int(item["revision"])
            if item_revision <= self._seen_phase_revision:
                continue
            phase = str(item["phase"])
            if phase != "READY" and (not self._phase_queue or self._phase_queue[-1] != phase):
                self._phase_queue.append(phase)
            self._seen_phase_revision = item_revision

    def poll(
        self,
        now: float | None = None,
        *,
        read_state: Callable[[Path], Any] | None = None,
    ) -> None:
        if not self.active:
            return
        assert self.process is not None
        current_time = time.monotonic() if now is None else float(now)
        return_code = self.process.poll()
        should_read = current_time >= self.next_poll_at or return_code is not None
        read_attempted = False
        if should_read:
            read_attempted = True
            self.next_poll_at = current_time + LIVE_STATE_POLL_SECONDS
            reader = _read_live_state if read_state is None else read_state
            try:
                self._accept_state(reader(self.config.live_state_path))
            except FileNotFoundError:
                self.warning = "WAITING FOR CONTROLLER LIVE STATE"
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                self.warning = f"LIVE STATE NOT YET VALID: {exc}"

        if return_code is None:
            return
        if not read_attempted:
            return
        if return_code != 0:
            detail = "controller process failed"
            if self.state is not None and self.state.error is not None:
                detail = str(self.state.error.get("message") or detail)
            self._fail(f"LIVE TRAINING FAILED: {detail}")
            return
        if self.state is None or self.state.phase != "GENERATION_COMPLETE":
            self._fail("LIVE TRAINING FAILED: controller exited without GENERATION_COMPLETE")
            return
        try:
            completed = _load_completed_live_generation(self.state, self.config)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self._fail(f"LIVE TRAINING FAILED: {exc}")
            return
        next_policy = self.state.next_policy_version
        next_generation = self.state.next_generation
        assert next_policy is not None and next_generation is not None
        self.current_policy_version = next_policy
        self.current_generation = next_generation
        self.current_checkpoint = completed[1]
        self._completed = completed
        self.process = None
        self.error = None
        self.locked_notice = False

    def _fail(self, message: str) -> None:
        self.error = message
        self.process = None
        self.locked_notice = False
        self._phase_queue.clear()
        self._display_phase = "ERROR"

    def take_completed_generation(self) -> list[AttemptTrail] | None:
        if self._completed is None:
            return None
        attempts, _ = self._completed
        self._completed = None
        return attempts

    def display_phase(self, now: float | None = None) -> str:
        current_time = time.monotonic() if now is None else float(now)
        if self._phase_queue and (
            self._display_phase is None or current_time >= self._display_phase_until
        ):
            self._display_phase = self._phase_queue.popleft()
            hold = 0.75 if self._display_phase in {"TRAINING_POLICY", "POLICY_UPDATED"} else 0.35
            self._display_phase_until = current_time + hold
        if self._display_phase is not None and current_time < self._display_phase_until:
            return self._display_phase
        if self.error is not None:
            return "ERROR"
        if self.state is not None:
            return self.state.phase
        return "READY"


def _xy(value: Any) -> tuple[float, float]:
    """Convert an environment position or velocity to a drawable pair."""

    return float(value[0]), float(value[1])


def _position(entity: dict[str, Any]) -> tuple[float, float]:
    return _xy(entity["position"])


def _screen_point(position: tuple[float, float]) -> tuple[float, float]:
    """Map a recorded world coordinate into the unobscured champion viewport."""

    scale = min(WORLD_VIEWPORT_WIDTH / WIDTH, WORLD_VIEWPORT_HEIGHT / HEIGHT)
    return (
        WORLD_VIEWPORT_X + position[0] * scale,
        WORLD_VIEWPORT_Y + position[1] * scale,
    )


def _screen_radius(radius: float) -> int:
    scale = min(WORLD_VIEWPORT_WIDTH / WIDTH, WORLD_VIEWPORT_HEIGHT / HEIGHT)
    return max(1, round(float(radius) * scale))


def make_stars(seed: int, count: int = 190) -> list[tuple[int, int, int, int, float, float]]:
    """Build a deterministic star field without changing the game state."""

    rng = random.Random((int(seed) * 1_000_003) ^ 0x5A17F13D)
    stars: list[tuple[int, int, int, int, float, float]] = []
    for _ in range(count):
        size_roll = rng.random()
        radius = 1 if size_roll < 0.78 else 2 if size_roll < 0.97 else 3
        stars.append(
            (
                rng.randrange(WIDTH),
                rng.randrange(HEIGHT),
                radius,
                rng.randint(115, 225),
                rng.random() * math.tau,
                rng.uniform(0.7, 2.0),
            )
        )
    return stars


def make_background(seed: int) -> tuple[Any, list[tuple[int, int, int, int, float, float]]]:
    """Create the static dark-space background for one seeded universe."""

    assert pygame is not None
    background = pygame.Surface((WIDTH, HEIGHT)).convert()
    background.fill(SPACE)

    nebula = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    rng = random.Random((int(seed) * 97_409) ^ 0xC05A05)
    colours = ((29, 47, 116), (82, 27, 112), (14, 78, 100))
    for index in range(8):
        center = (rng.randrange(-120, WIDTH + 120), rng.randrange(-100, HEIGHT + 100))
        radius = rng.randrange(140, 310)
        colour = colours[index % len(colours)]
        for layer in range(6, 0, -1):
            pygame.draw.circle(
                nebula,
                (*colour, 4 + 6 - layer),
                center,
                int(radius * layer / 6),
            )
    background.blit(nebula, (0, 0))
    return background, make_stars(seed)


def draw_stars(
    screen: Any,
    stars: list[tuple[int, int, int, int, float, float]],
    elapsed: float,
) -> None:
    assert pygame is not None
    for x, y, radius, base, phase, speed in stars:
        brightness = max(80, min(255, int(base + 28 * math.sin(elapsed * speed + phase))))
        colour = (brightness, min(255, brightness + 7), min(255, brightness + 24))
        pygame.draw.circle(screen, colour, (x, y), radius)
        if radius == 3:
            faint = (brightness // 2, brightness // 2, min(255, brightness // 2 + 25))
            pygame.draw.line(screen, faint, (x - 5, y), (x + 5, y))
            pygame.draw.line(screen, faint, (x, y - 5), (x, y + 5))


def draw_trail(screen: Any, trail: deque[tuple[float, float]]) -> None:
    """Draw old positions dimly and recent positions brightly."""

    assert pygame is not None
    if len(trail) < 2:
        return

    points = [_screen_point(point) for point in trail]
    count = len(points)
    for index in range(1, count):
        freshness = index / count
        colour = (
            int(7 + 38 * freshness),
            int(22 + 186 * freshness),
            int(45 + 210 * freshness),
        )
        width = 1 if freshness < 0.55 else 2 if freshness < 0.88 else 3
        pygame.draw.line(screen, colour, points[index - 1], points[index], width)


def _ghost_attempts(
    attempts: list[AttemptTrail],
    current_seed: int,
    active_attempt: AttemptTrail | None = None,
) -> list[AttemptTrail]:
    """Select earlier paths without leaking the active or another universe."""

    candidates = [
        attempt
        for attempt in attempts
        if attempt is not active_attempt
        and attempt.seed == current_seed
        and len(attempt.points) >= 2
    ]
    if active_attempt is None:
        return candidates
    if active_attempt.generation is None:
        return []
    return [
        attempt
        for attempt in candidates
        if attempt.generation is not None
        and attempt.generation < active_attempt.generation
    ]


def draw_ghost_trails(
    screen: Any,
    attempts: list[AttemptTrail],
    current_seed: int,
    active_attempt: AttemptTrail | None = None,
) -> None:
    """Render earlier paths only when they belong to this exact universe.

    A trajectory from another seed is never projected over the active world's
    geometry.  The active replay is also excluded so its future path is not
    revealed before the replay cursor reaches it.
    """

    assert pygame is not None
    eligible = _ghost_attempts(attempts, current_seed, active_attempt)
    if not eligible:
        return

    layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for attempt in eligible:
        colour = (89, 255, 183) if attempt.is_champion else (115, 139, 185)
        generation_age = (
            max(1, active_attempt.generation - attempt.generation)
            if active_attempt is not None
            and active_attempt.generation is not None
            and attempt.generation is not None
            else 1
        )
        alpha = max(20, (82 if attempt.is_champion else 42) - generation_age * 7)
        width = 2 if attempt.is_champion and generation_age <= 2 else 1
        screen_points = [_screen_point(point) for point in attempt.points]
        pygame.draw.lines(layer, (*colour, alpha), False, screen_points, width)

        if attempt.is_champion:
            endpoint = tuple(round(value) for value in screen_points[-1])
            pygame.draw.circle(layer, (*colour, 145), endpoint, 6, 2)

    screen.blit(layer, (0, 0))


def draw_recorded_trail(screen: Any, attempt: AttemptTrail, cursor: int) -> None:
    """Reveal the exact recorded points with a long, layered glow."""

    assert pygame is not None
    end = min(len(attempt.points), max(1, cursor + 1))
    start = max(0, end - TRAIL_LENGTH)
    points = tuple(_screen_point(point) for point in attempt.points[start:end])
    if len(points) < 2:
        return

    colour = GREEN if attempt.is_champion else CYAN
    glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.lines(glow, (*colour, 22), False, points, 11)
    pygame.draw.lines(glow, (*colour, 58), False, points, 5)
    screen.blit(glow, (0, 0))

    count = len(points)
    for index in range(1, count):
        freshness = index / count
        segment_colour = tuple(
            int(channel * (0.27 + freshness * 0.73)) for channel in colour
        )
        width = 2 if freshness < 0.72 else 3
        pygame.draw.line(
            screen,
            segment_colour,
            points[index - 1],
            points[index],
            width,
        )


def _attempt_group(
    attempts: list[AttemptTrail], active: AttemptTrail
) -> list[AttemptTrail]:
    grouped = [
        attempt
        for attempt in attempts
        if attempt.generation == active.generation
        and attempt.policy_version == active.policy_version
    ]
    return grouped or [active]


def _outcome_label(attempt: AttemptTrail) -> str:
    if attempt.success is True:
        return "SUCCESS"
    if attempt.success is False:
        return (attempt.termination or "FAILURE").replace("_", " ").upper()
    return (attempt.termination or "OUTCOME N/A").replace("_", " ").upper()


def _outcome_colour(attempt: AttemptTrail) -> tuple[int, int, int]:
    if attempt.success is True:
        return GREEN
    termination = (attempt.termination or "").lower()
    if "collision" in termination:
        return RED
    if termination in {"timeout", "out_of_bounds"}:
        return (255, 190, 74)
    return MUTED


def _short_identifier(value: str | None, limit: int = 24) -> str:
    """Return a compact, unchanged-at-the-ends identifier for small cards."""

    if value is None:
        return "N/A"
    if len(value) <= limit:
        return value
    side = max(3, (limit - 1) // 2)
    return f"{value[:side]}…{value[-side:]}"


def _lifecycle_states(attempt: AttemptTrail) -> tuple[str, ...]:
    """Return the exact recorded lifecycle states in artifact order."""

    return tuple(
        str(event["state"]).upper()
        for event in attempt.lifecycle
        if isinstance(event.get("state"), str) and event["state"]
    )


def _learning_history(attempts: list[AttemptTrail]) -> list[dict[str, Any]]:
    """Summarize real loaded generations without assuming improvement."""

    grouped: dict[tuple[int, int | None], list[AttemptTrail]] = {}
    for attempt in attempts:
        if attempt.generation is None:
            continue
        grouped.setdefault((attempt.generation, attempt.policy_version), []).append(attempt)

    history: list[dict[str, Any]] = []
    for (generation, policy_version), group in sorted(grouped.items()):
        rewards = [float(item.reward) for item in group if item.reward is not None]
        outcomes = [bool(item.success) for item in group if item.success is not None]
        collisions = [
            "collision" in str(item.termination or "").lower()
            for item in group
            if item.termination is not None
        ]
        first = group[0]
        average_reward = first.generation_average_reward
        if average_reward is None and rewards:
            average_reward = math.fsum(rewards) / len(rewards)
        best_reward = first.generation_best_reward
        if best_reward is None and rewards:
            best_reward = max(rewards)
        success_rate = first.generation_success_rate
        if success_rate is None and outcomes:
            success_rate = sum(outcomes) / len(outcomes)
        collision_rate = first.generation_collision_rate
        if collision_rate is None and collisions:
            collision_rate = sum(collisions) / len(collisions)
        history.append(
            {
                "generation": generation,
                "policy_version": policy_version,
                "average_reward": average_reward,
                "best_reward": best_reward,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "world_count": first.generation_world_count or len(group),
                "daytona_verified": all(item.daytona_verified for item in group),
            }
        )
    return history


def _mini_world_surface(
    attempt: AttemptTrail,
    env: GravityEnv,
    *,
    active: bool,
) -> Any:
    """Draw one path over its own deterministic universe—not another seed."""

    assert pygame is not None
    width, height = 104, 55
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    surface.fill((3, 7, 20, 238))

    def scaled(position: tuple[float, float]) -> tuple[int, int]:
        return (
            round(position[0] * width / env.width),
            round(position[1] * height / env.height),
        )

    for planet in env.planets:
        center = scaled(_position(planet))
        radius = max(2, round(float(planet["radius"]) * width / env.width))
        gravity_radius = max(
            radius + 2,
            round(float(planet.get("gravity_radius", planet["radius"] * 4)) * width / env.width),
        )
        colour = tuple(planet.get("colour", planet.get("color", (92, 175, 255))))
        pygame.draw.circle(surface, (*colour, 60), center, gravity_radius, 1)
        pygame.draw.circle(surface, colour, center, radius)
    for asteroid in env.asteroids:
        pygame.draw.circle(
            surface,
            (119, 126, 145),
            scaled(_position(asteroid)),
            max(1, round(float(asteroid["radius"]) * width / env.width)),
        )
    portal_center = scaled(_position(env.portal))
    pygame.draw.circle(surface, (80, 248, 255), portal_center, 4, 1)
    pygame.draw.circle(surface, (80, 248, 255), portal_center, 1)

    path = [scaled(point) for point in attempt.points]
    if len(path) >= 2:
        path_colour = GREEN if attempt.is_champion else CYAN if active else _outcome_colour(attempt)
        pygame.draw.lines(surface, (*path_colour, 65), False, path, 5)
        pygame.draw.lines(surface, path_colour, False, path, 2 if active else 1)
        pygame.draw.circle(surface, _outcome_colour(attempt), path[-1], 3)
    return surface


def draw_parallel_universes(
    screen: Any,
    attempts: list[AttemptTrail],
    active: AttemptTrail,
    preview_envs: dict[int, GravityEnv],
    fonts: tuple[Any, Any, Any],
) -> None:
    """Show up to eight seed-correct universe/trajectory cards at once."""

    assert pygame is not None
    _, text_font, small_font = fonts
    group = sorted(
        _attempt_group(attempts, active),
        key=lambda attempt: (
            attempt.world_index is None,
            attempt.world_index if attempt.world_index is not None else 0,
            attempt.seed if attempt.seed is not None else 0,
        ),
    )[:8]
    panel = pygame.Rect(858, 18, 324, 704)
    translucent_panel(screen, panel)
    heading = (
        "PARALLEL DAYTONA UNIVERSES"
        if active.daytona_verified
        else "LOCAL PREVIEW UNIVERSES"
        if all(item.provenance == "LOCAL PREVIEW" for item in group)
        else "PARALLEL RECORDED UNIVERSES"
    )
    screen.blit(text_font.render(heading, True, WHITE), (872, 29))
    policy = f"v{active.policy_version}" if active.policy_version is not None else "vN/A"
    generation = f"G{active.generation}" if active.generation is not None else "GN/A"
    trained_next = (
        f" → v{active.next_policy_version}"
        if active.daytona_verified and active.next_policy_version is not None
        else ""
    )
    world_count = active.generation_world_count or len(group)
    screen.blit(
        small_font.render(
            f"{generation}  •  {policy}{trained_next}  •  {world_count} WORLDS",
            True,
            MUTED,
        ),
        (872, 52),
    )

    card_y = 78
    for index, attempt in enumerate(group):
        selected = attempt is active
        card = pygame.Rect(869, card_y + index * 79, 302, 72)
        border = GREEN if attempt.is_champion else CYAN if selected else (48, 76, 111)
        card_layer = pygame.Surface(card.size, pygame.SRCALPHA)
        pygame.draw.rect(
            card_layer,
            (8, 20, 43, 235 if selected or attempt.is_champion else 205),
            card_layer.get_rect(),
            border_radius=8,
        )
        pygame.draw.rect(
            card_layer,
            (*border, 220),
            card_layer.get_rect(),
            2 if selected or attempt.is_champion else 1,
            border_radius=8,
        )
        screen.blit(card_layer, card.topleft)

        if attempt.seed is not None:
            preview = _mini_world_surface(
                attempt,
                preview_envs[attempt.seed],
                active=selected,
            )
            screen.blit(preview, (card.x + 7, card.y + 8))

        world_number = attempt.world_index if attempt.world_index is not None else index + 1
        if attempt.is_champion:
            badge = (
                " • CHAMPION"
                if attempt.daytona_verified
                else " • PREVIEW BEST"
                if attempt.provenance == "LOCAL PREVIEW"
                else " • RECORDED BEST"
            )
        else:
            badge = " • ACTIVE" if selected else ""
        reward = f"{attempt.reward:+.2f}" if attempt.reward is not None else "N/A"
        outcome = _outcome_label(attempt)
        card_outcome = (
            "COLLISION"
            if "COLLISION" in outcome
            else "OUT BOUNDS"
            if "OUT OF BOUNDS" in outcome
            else outcome[:11]
        )
        states = _lifecycle_states(attempt)
        if attempt.daytona_verified:
            execution = (
                "DAYTONA • FINISHED"
                if "RESULT_COLLECTED" in states or not states
                else "DAYTONA • RECORDED"
            )
        elif attempt.provenance == "LOCAL PREVIEW":
            execution = "LOCAL PREVIEW"
        else:
            execution = "UNVERIFIED REPLAY"
        screen.blit(
            small_font.render(f"WORLD {world_number:02d}{badge}", True, border),
            (card.x + 119, card.y + 5),
        )
        screen.blit(
            small_font.render(f"RWD {reward} • {card_outcome}", True, _outcome_colour(attempt)),
            (card.x + 119, card.y + 20),
        )
        screen.blit(
            small_font.render(execution, True, GREEN if attempt.daytona_verified else MUTED),
            (card.x + 119, card.y + 36),
        )
        screen.blit(
            small_font.render(
                f"ID {_short_identifier(attempt.sandbox_id, 20)}",
                True,
                WHITE if attempt.sandbox_id else MUTED,
            ),
            (card.x + 119, card.y + 52),
        )


def draw_replay_banner(
    screen: Any,
    attempt: AttemptTrail,
    fonts: tuple[Any, Any, Any],
) -> None:
    assert pygame is not None
    title_font, text_font, small_font = fonts
    rect = pygame.Rect(18, 18, 482, 82)
    translucent_panel(screen, rect)
    if attempt.daytona_verified and attempt.is_champion:
        headline = "CURRENT CHAMPION"
        colour = GREEN
    elif attempt.daytona_verified:
        world = attempt.world_index if attempt.world_index is not None else "?"
        headline = f"PARALLEL WORLD {world}"
        colour = CYAN
    elif attempt.provenance == "LOCAL PREVIEW":
        headline = "LOCAL PREVIEW"
        colour = (255, 190, 74)
    else:
        headline = "UNVERIFIED REPLAY"
        colour = (255, 190, 74)
    reward = f"{attempt.reward:+.2f}" if attempt.reward is not None else "N/A"
    world = attempt.world_index if attempt.world_index is not None else "?"
    generation = attempt.generation if attempt.generation is not None else "N/A"
    policy = f"v{attempt.policy_version}" if attempt.policy_version is not None else "N/A"
    outcome = _outcome_label(attempt)
    if "COLLISION" in outcome:
        outcome = "COLLISION"
    elif "OUT OF BOUNDS" in outcome:
        outcome = "OUT BOUNDS"
    screen.blit(title_font.render("GRAVITY GAUNTLET", True, WHITE), (34, 28))
    badge = text_font.render(headline, True, colour)
    screen.blit(badge, (rect.right - badge.get_width() - 18, 34))
    detail = (
        f"WORLD {world}  •  GEN {generation}  •  {policy}  •  "
        f"{outcome}  •  RWD {reward}"
    )
    screen.blit(small_font.render(detail, True, WHITE), (35, 67))


def _lifecycle_colour(state: str) -> tuple[int, int, int]:
    if state in {"SUCCESS", "RESULT_COLLECTED"}:
        return GREEN
    if state in {"COLLISION", "ERROR"}:
        return RED
    if state in {"OUT_OF_BOUNDS", "TIMEOUT"}:
        return (255, 190, 74)
    if state in {"LIVE", "RUNNING"}:
        return CYAN
    return MUTED


def draw_lifecycle_panel(
    screen: Any,
    attempt: AttemptTrail,
    fonts: tuple[Any, Any, Any],
) -> None:
    """Show recorded sandbox events as evidence, never as a simulated live feed."""

    assert pygame is not None
    _, text_font, small_font = fonts
    panel = pygame.Rect(520, 18, 320, 82)
    translucent_panel(screen, panel)
    title = (
        "RECORDED DAYTONA LIFECYCLE"
        if attempt.daytona_verified
        else "EXECUTION PROVENANCE"
    )
    screen.blit(text_font.render(title, True, WHITE), (536, 29))
    states = _lifecycle_states(attempt)
    if not states:
        if attempt.provenance == "LOCAL PREVIEW":
            primary = "LOCAL PREVIEW — NO DAYTONA CLAIM"
        else:
            primary = "NO LIFECYCLE EVENTS IN ARTIFACT"
        screen.blit(small_font.render(primary, True, (255, 190, 74)), (536, 65))
        return

    shown = states[:5]
    for index, state in enumerate(shown):
        x = 540 + index * 59
        colour = _lifecycle_colour(state)
        if index:
            pygame.draw.line(screen, (55, 94, 125), (x - 21, 67), (x - 6, 67), 2)
        pygame.draw.circle(screen, colour, (x, 67), 4)
        detail = {
            "CREATING": "CREATE",
            "RUNNING": "RUN",
            "OUT_OF_BOUNDS": "OUT",
            "RESULT_COLLECTED": "COLLECT",
        }.get(state, state[:7])
        label = small_font.render(detail, True, colour)
        screen.blit(label, label.get_rect(center=(x, 52)))


def draw_learning_history_panel(
    screen: Any,
    attempts: list[AttemptTrail],
    fonts: tuple[Any, Any, Any],
) -> None:
    """Plot actual generation rewards; a falling line remains a falling line."""

    assert pygame is not None
    history = _learning_history(attempts)
    if len(history) < 2:
        return
    _, text_font, small_font = fonts
    panel = pygame.Rect(520, 625, 320, 112)
    translucent_panel(screen, panel)
    verified = all(bool(item["daytona_verified"]) for item in history)
    local_preview = all(
        attempt.provenance == "LOCAL PREVIEW" for attempt in attempts
    )
    title = (
        "REAL DAYTONA LEARNING HISTORY"
        if verified
        else "LOCAL PREVIEW HISTORY"
        if local_preview
        else "UNVERIFIED RECORDED HISTORY"
    )
    screen.blit(text_font.render(title, True, WHITE), (536, 634))
    screen.blit(
        small_font.render("AVG REWARD", True, CYAN),
        (536, 658),
    )
    screen.blit(
        small_font.render("BEST", True, GREEN),
        (626, 658),
    )

    plot = pygame.Rect(536, 678, 286, 26)
    values = [
        float(value)
        for item in history
        for value in (item["average_reward"], item["best_reward"])
        if value is not None
    ]
    if not values:
        screen.blit(small_font.render("NO REWARD METRICS IN ARTIFACT", True, MUTED), plot.topleft)
        return
    low, high = min(values), max(values)
    padding = max(1.0, (high - low) * 0.08)
    low -= padding
    high += padding

    def point(index: int, value: float) -> tuple[int, int]:
        x = plot.x + round(index * plot.width / max(1, len(history) - 1))
        ratio = (value - low) / (high - low)
        return x, plot.bottom - round(ratio * plot.height)

    average_points = [
        point(index, float(item["average_reward"]))
        for index, item in enumerate(history)
        if item["average_reward"] is not None
    ]
    best_points = [
        point(index, float(item["best_reward"]))
        for index, item in enumerate(history)
        if item["best_reward"] is not None
    ]
    if len(average_points) >= 2:
        pygame.draw.lines(screen, CYAN, False, average_points, 2)
    if len(best_points) >= 2:
        pygame.draw.lines(screen, GREEN, False, best_points, 2)
    for colour, points in ((CYAN, average_points), (GREEN, best_points)):
        for chart_point in points:
            pygame.draw.circle(screen, colour, chart_point, 4)

    label_indices = range(len(history)) if len(history) <= 8 else range(0, len(history), 2)
    for index in label_indices:
        item = history[index]
        x = plot.x + round(index * plot.width / max(1, len(history) - 1))
        label = small_font.render(f"G{item['generation']}", True, MUTED)
        screen.blit(label, label.get_rect(center=(x, 721)))


def draw_speed_streaks(
    screen: Any,
    position: tuple[float, float],
    velocity: tuple[float, float],
) -> None:
    """Add restrained motion cues derived only from the real ship velocity."""

    assert pygame is not None
    speed = math.hypot(*velocity)
    if speed < 60.0:
        return
    position = _screen_point(position)
    direction = (velocity[0] / speed, velocity[1] / speed)
    side = (-direction[1], direction[0])
    length = min(34.0, 5.0 + speed * 0.065)
    alpha = max(25, min(125, int((speed - 60.0) * 0.7)))
    layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for offset, scale in ((-8.0, 0.55), (0.0, 1.0), (8.0, 0.55)):
        start = (
            position[0] + side[0] * offset - direction[0] * 15.0,
            position[1] + side[1] * offset - direction[1] * 15.0,
        )
        end = (
            start[0] - direction[0] * length * scale,
            start[1] - direction[1] * length * scale,
        )
        pygame.draw.line(layer, (92, 222, 255, int(alpha * scale)), start, end, 2)
    screen.blit(layer, (0, 0))


def _gravity_ring(
    layer: Any,
    center: tuple[int, int],
    radius: int,
    colour: tuple[int, int, int],
) -> None:
    assert pygame is not None
    pygame.draw.circle(layer, (*colour, 32), center, radius, 1)
    pygame.draw.circle(layer, (*colour, 17), center, max(2, int(radius * 0.62)), 1)

    rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
    for segment in range(12):
        start = segment * math.tau / 12 + 0.05
        pygame.draw.arc(layer, (*colour, 75), rect, start, start + 0.28, 2)


def draw_planets(screen: Any, planets: list[dict[str, Any]], seed: int) -> None:
    """Draw gravity influence first, then bright spherical planet bodies."""

    assert pygame is not None
    effects = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    for index, planet in enumerate(planets):
        x, y = _position(planet)
        center = tuple(round(value) for value in _screen_point((x, y)))
        radius = max(3, _screen_radius(float(planet["radius"])))
        fallback = PLANET_PALETTE[(seed + index * 3) % len(PLANET_PALETTE)]
        colour = tuple(planet.get("colour", planet.get("color", fallback)))
        gravity_radius = max(
            radius + 2,
            _screen_radius(
                float(planet.get("gravity_radius", float(planet["radius"]) * 4))
            ),
        )
        _gravity_ring(effects, center, gravity_radius, colour)
        pygame.draw.circle(effects, (*colour, 18), center, max(radius + 18, int(radius * 1.7)))
        pygame.draw.circle(effects, (*colour, 38), center, radius + 10)

    screen.blit(effects, (0, 0))

    for index, planet in enumerate(planets):
        x, y = _position(planet)
        center = tuple(round(value) for value in _screen_point((x, y)))
        radius = max(3, _screen_radius(float(planet["radius"])))
        fallback = PLANET_PALETTE[(seed + index * 3) % len(PLANET_PALETTE)]
        colour = tuple(planet.get("colour", planet.get("color", fallback)))

        pygame.draw.circle(screen, (7, 9, 21), (center[0] + 4, center[1] + 6), radius + 2)
        pygame.draw.circle(screen, colour, center, radius)
        highlight = tuple(min(255, channel + 100) for channel in colour)
        highlight_center = (center[0] - int(radius * 0.27), center[1] - int(radius * 0.30))
        for layer in range(4, 0, -1):
            local_radius = max(2, int(radius * layer / 8))
            blend = layer / 7
            layer_colour = tuple(
                int(colour[channel] * (1 - blend) + highlight[channel] * blend)
                for channel in range(3)
            )
            pygame.draw.circle(screen, layer_colour, highlight_center, local_radius)


def asteroid_points(
    center: tuple[float, float], radius: float, seed: int, index: int
) -> list[tuple[int, int]]:
    rng = random.Random((int(seed) + 31) * 65_537 + index * 8_191)
    count = rng.randint(8, 11)
    points: list[tuple[int, int]] = []
    for point_index in range(count):
        angle = point_index * math.tau / count
        local_radius = radius * rng.uniform(0.76, 1.12)
        points.append(
            (
                round(center[0] + math.cos(angle) * local_radius),
                round(center[1] + math.sin(angle) * local_radius),
            )
        )
    return points


def draw_asteroids(screen: Any, asteroids: list[dict[str, Any]], seed: int) -> None:
    assert pygame is not None
    for index, asteroid in enumerate(asteroids):
        center = _screen_point(_position(asteroid))
        radius = float(_screen_radius(float(asteroid["radius"])))
        points = asteroid_points(center, radius, seed, index)
        shadow = [(x + 3, y + 4) for x, y in points]
        base = tuple(asteroid.get("colour", asteroid.get("color", (91, 98, 114))))

        pygame.draw.polygon(screen, (12, 15, 26), shadow)
        pygame.draw.polygon(screen, base, points)
        pygame.draw.lines(screen, (160, 170, 190), True, points, 2)

        x, y = round(center[0]), round(center[1])
        crater = max(2, int(radius * 0.19))
        pygame.draw.circle(
            screen,
            (54, 60, 75),
            (x - int(radius * 0.25), y - int(radius * 0.16)),
            crater,
        )


def draw_portal(screen: Any, portal: dict[str, Any], elapsed: float) -> None:
    assert pygame is not None
    x, y = _position(portal)
    center = tuple(round(value) for value in _screen_point((x, y)))
    radius = max(5, _screen_radius(float(portal["radius"])))
    pulse = 1.0 + 0.08 * math.sin(elapsed * 4.2)
    colour = tuple(portal.get("colour", portal.get("color", (80, 245, 255))))

    glow_radius = int(radius * 3.7 * pulse)
    glow = pygame.Surface((glow_radius * 2 + 4, glow_radius * 2 + 4), pygame.SRCALPHA)
    local_center = (glow.get_width() // 2, glow.get_height() // 2)
    for layer in range(7, 0, -1):
        pygame.draw.circle(
            glow,
            (*colour, 5 + (8 - layer) * 5),
            local_center,
            max(radius, int(glow_radius * layer / 7)),
        )
    screen.blit(glow, (center[0] - local_center[0], center[1] - local_center[1]))

    outer = int(radius * 1.55 * pulse)
    pygame.draw.circle(screen, (25, 98, 135), center, outer, max(2, radius // 7))
    pygame.draw.circle(screen, colour, center, radius, max(3, radius // 6))
    pygame.draw.circle(screen, (3, 9, 27), center, max(2, int(radius * 0.67)))
    pygame.draw.circle(screen, (174, 254, 255), center, max(2, int(radius * 0.39)), 2)

    rect = pygame.Rect(center[0] - outer, center[1] - outer, outer * 2, outer * 2)
    for index in range(3):
        start = elapsed * (1.1 + index * 0.18) + index * math.tau / 3
        pygame.draw.arc(screen, (155, 250, 255), rect, start, start + 0.82, 3)


def draw_ship(
    screen: Any,
    position: tuple[float, float],
    angle: float,
    thrust: tuple[float, float],
    ship_radius: float,
) -> None:
    assert pygame is not None
    x, y = _screen_point(position)
    radius = max(7.0, _screen_radius(ship_radius) * 1.35)
    forward = (math.cos(angle), math.sin(angle))
    side = (-forward[1], forward[0])

    def point(longitudinal: float, lateral: float) -> tuple[int, int]:
        return (
            round(x + forward[0] * longitudinal + side[0] * lateral),
            round(y + forward[1] * longitudinal + side[1] * lateral),
        )

    glow_radius = round(radius * 2.5)
    glow = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow, (68, 207, 255, 28), (glow_radius, glow_radius), glow_radius)
    screen.blit(glow, (round(x) - glow_radius, round(y) - glow_radius))

    thrust_magnitude = math.hypot(*thrust)
    if thrust_magnitude > 0.0:
        thrust_direction = (thrust[0] / thrust_magnitude, thrust[1] / thrust_magnitude)
        exhaust_side = (-thrust_direction[1], thrust_direction[0])
        flame_length = radius * (1.35 + 0.20 * math.sin(pygame.time.get_ticks() * 0.04))
        base_x = x - thrust_direction[0] * radius * 0.55
        base_y = y - thrust_direction[1] * radius * 0.55
        flame = [
            (
                round(base_x + exhaust_side[0] * radius * 0.25),
                round(base_y + exhaust_side[1] * radius * 0.25),
            ),
            (
                round(x - thrust_direction[0] * flame_length),
                round(y - thrust_direction[1] * flame_length),
            ),
            (
                round(base_x - exhaust_side[0] * radius * 0.25),
                round(base_y - exhaust_side[1] * radius * 0.25),
            ),
        ]
        pygame.draw.polygon(screen, (255, 97, 50), flame)

    hull = [
        point(radius, 0),
        point(-radius * 0.68, -radius * 0.72),
        point(-radius * 0.34, 0),
        point(-radius * 0.68, radius * 0.72),
    ]
    pygame.draw.polygon(screen, WHITE, hull)
    pygame.draw.lines(screen, (62, 191, 255), True, hull, 2)
    pygame.draw.circle(screen, (144, 239, 255), point(radius * 0.28, 0), max(2, int(radius * 0.18)))


def keyboard_action() -> tuple[float, float]:
    assert pygame is not None
    keys = pygame.key.get_pressed()
    thrust_x = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(
        keys[pygame.K_a] or keys[pygame.K_LEFT]
    )
    thrust_y = float(keys[pygame.K_s] or keys[pygame.K_DOWN]) - float(
        keys[pygame.K_w] or keys[pygame.K_UP]
    )
    magnitude = math.hypot(thrust_x, thrust_y)
    if magnitude > 1.0:
        thrust_x /= magnitude
        thrust_y /= magnitude
    return thrust_x, thrust_y


def distance_to_portal(env: GravityEnv) -> float:
    ship_x, ship_y = _xy(env.ship_position)
    portal_x, portal_y = _position(env.portal)
    return math.hypot(portal_x - ship_x, portal_y - ship_y)


def status_text(env: GravityEnv) -> tuple[str, tuple[int, int, int]]:
    status = str(env.status).replace("_", " ").upper()
    if env.success:
        return "PORTAL REACHED", GREEN
    if env.done:
        if "TIME" in status:
            return "TIME LIMIT", (255, 201, 89)
        return status or "SHIP LOST", RED
    return "IN FLIGHT", CYAN


def translucent_panel(screen: Any, rect: Any) -> None:
    assert pygame is not None
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (5, 12, 31, 216), panel.get_rect(), border_radius=12)
    pygame.draw.rect(panel, (55, 117, 169, 120), panel.get_rect(), 1, border_radius=12)
    screen.blit(panel, rect.topleft)


def draw_hud(
    screen: Any,
    env: GravityEnv,
    seed: int,
    fonts: tuple[Any, ...],
    attempts: list[AttemptTrail],
    *,
    generation: int | None,
    policy_version: int | None,
    source_label: str,
) -> None:
    """Draw an unmistakably local-only HUD without obscuring the flight viewport."""

    assert pygame is not None
    title_font, text_font, small_font = fonts
    del generation, policy_version
    vx, vy = _xy(env.ship_velocity)
    speed = math.hypot(vx, vy)
    status, status_colour = status_text(env)
    completed_rewards = [
        float(attempt.reward) for attempt in attempts if attempt.reward is not None
    ]
    average_reward = (
        f"{sum(completed_rewards) / len(completed_rewards):8.2f}"
        if completed_rewards
        else "     N/A"
    )
    best_reward = f"{max(completed_rewards):8.2f}" if completed_rewards else "     N/A"
    known_outcomes = [attempt.success for attempt in attempts if attempt.success is not None]
    success_rate = (
        f"{100.0 * sum(bool(outcome) for outcome in known_outcomes) / len(known_outcomes):6.1f}%"
        if known_outcomes
        else "   N/A"
    )
    info = env.info()
    clearance_seen = float(info["min_clearance_seen"])
    banner = pygame.Rect(18, 18, 482, 82)
    translucent_panel(screen, banner)
    screen.blit(title_font.render("GRAVITY GAUNTLET", True, WHITE), (34, 29))
    preview_badge = text_font.render("LOCAL PREVIEW", True, (255, 190, 74))
    screen.blit(preview_badge, (banner.right - preview_badge.get_width() - 18, 35))
    screen.blit(
        small_font.render(
            f"SEED {seed}  •  MANUAL CONTROL  •  {status}",
            True,
            status_colour,
        ),
        (35, 67),
    )

    provenance = pygame.Rect(520, 18, 320, 82)
    translucent_panel(screen, provenance)
    screen.blit(text_font.render("EXECUTION PROVENANCE", True, WHITE), (536, 29))
    screen.blit(
        small_font.render("LOCAL PREVIEW — NO DAYTONA CLAIM", True, (255, 190, 74)),
        (536, 54),
    )
    screen.blit(
        small_font.render("KEYBOARD PHYSICS ONLY • NO SANDBOX", True, MUTED),
        (536, 72),
    )

    panel = pygame.Rect(18, 625, 822, 112)
    translucent_panel(screen, panel)
    screen.blit(
        text_font.render("LOCAL PHYSICS PREVIEW  •  MANUAL CONTROL", True, WHITE),
        (34, 636),
    )
    rows = (
        (
            (34, f"CURRENT RWD {env.episode_reward:+.2f}", WHITE),
            (210, f"RUN AVG {average_reward.strip()}", CYAN),
            (366, f"RUN BEST {best_reward.strip()}", GREEN),
            (531, f"SUCCESS {success_rate.strip()} / {len(known_outcomes)}", WHITE),
        ),
        (
            (34, f"VELOCITY ({vx:+.1f}, {vy:+.1f})", WHITE),
            (236, f"SPEED {speed:.2f}", WHITE),
            (370, f"TARGET {distance_to_portal(env):.1f}px", WHITE),
            (
                531,
                f"MIN CLEAR {clearance_seen:.2f}px",
                WHITE if clearance_seen >= 58 else RED,
            ),
        ),
        (
            (34, f"SEED {seed}", WHITE),
            (210, f"STEP {env.timestep}", WHITE),
            (366, f"STATUS {status}", status_colour),
            (600, f"SOURCE {source_label}", MUTED),
        ),
    )
    for row_y, row in zip((661, 683, 705), rows):
        for x, text, colour in row:
            screen.blit(small_font.render(text, True, colour), (x, row_y))

    controls = pygame.Rect(18, HEIGHT - 61, 510, 43)
    translucent_panel(screen, controls)
    hint = "WASD / ARROWS  THRUST     R  RESTART     N  NEW UNIVERSE"
    screen.blit(text_font.render(hint, True, MUTED), (34, HEIGHT - 49))


def _recorded_velocity(attempt: AttemptTrail, cursor: int) -> tuple[float, float] | None:
    if not attempt.trajectory:
        return None
    sample = attempt.trajectory[min(cursor, len(attempt.trajectory) - 1)]
    if sample.get("vx") is None or sample.get("vy") is None:
        return None
    return float(sample["vx"]), float(sample["vy"])


def _recorded_clearance(attempt: AttemptTrail, cursor: int) -> float | None:
    if not attempt.trajectory:
        return None
    sample = attempt.trajectory[min(cursor, len(attempt.trajectory) - 1)]
    return (
        float(sample["clearance"])
        if sample.get("clearance") is not None
        else None
    )


def _recorded_thrust(attempt: AttemptTrail, cursor: int) -> tuple[float, float]:
    """Return the recorded policy vector for this physics point, or no flame."""

    if not attempt.action_vectors or not attempt.trajectory or cursor <= 0:
        return 0.0, 0.0
    sample = attempt.trajectory[min(cursor, len(attempt.trajectory) - 1)]
    step_value = sample.get("step")
    step = int(step_value) if step_value is not None else cursor
    decision_index = max(0, (step - 1) // ACTION_HOLD_STEPS)
    if decision_index >= len(attempt.action_vectors):
        return 0.0, 0.0
    return attempt.action_vectors[decision_index]


def _recorded_reward_so_far(attempt: AttemptTrail, cursor: int) -> float | None:
    if not attempt.trajectory:
        return None
    rewards = [
        float(point["reward"])
        for point in attempt.trajectory[: cursor + 1]
        if point.get("reward") is not None
    ]
    return math.fsum(rewards) if rewards else None


def draw_replay_hud(
    screen: Any,
    attempt: AttemptTrail,
    attempts: list[AttemptTrail],
    cursor: int,
    fonts: tuple[Any, Any, Any],
) -> None:
    """Show only metrics carried by, or derived from, the selected recording."""

    assert pygame is not None
    _, text_font, small_font = fonts
    group = _attempt_group(attempts, attempt)
    scored = [float(item.reward) for item in group if item.reward is not None]
    outcomes = [item.success for item in group if item.success is not None]
    collisions = [
        "collision" in str(item.termination or "").lower()
        for item in group
        if item.termination is not None
    ]
    velocity = _recorded_velocity(attempt, cursor)
    speed = math.hypot(*velocity) if velocity is not None else None
    clearance = _recorded_clearance(attempt, cursor)
    reward_so_far = _recorded_reward_so_far(attempt, cursor)
    finished = cursor >= len(attempt.points) - 1
    status = _outcome_label(attempt) if finished else "RECORDED FLIGHT IN PROGRESS"
    status_colour = _outcome_colour(attempt) if finished else CYAN
    sandbox_id = attempt.sandbox_id or "N/A"
    if len(sandbox_id) > 44:
        sandbox_id = f"{sandbox_id[:20]}…{sandbox_id[-20:]}"
    reward_text = f"{attempt.reward:+.2f}" if attempt.reward is not None else "N/A"
    current_reward_text = f"{reward_so_far:+.2f}" if reward_so_far is not None else "N/A"
    average_value = (
        attempt.generation_average_reward
        if attempt.generation_average_reward is not None
        else math.fsum(scored) / len(scored)
        if scored
        else None
    )
    best_value = (
        attempt.generation_best_reward
        if attempt.generation_best_reward is not None
        else max(scored)
        if scored
        else None
    )
    success_value = (
        attempt.generation_success_rate
        if attempt.generation_success_rate is not None
        else sum(bool(value) for value in outcomes) / len(outcomes)
        if outcomes
        else None
    )
    collision_value = (
        attempt.generation_collision_rate
        if attempt.generation_collision_rate is not None
        else sum(collisions) / len(collisions)
        if collisions
        else None
    )
    average_reward = f"{average_value:+.2f}" if average_value is not None else "N/A"
    best_reward = f"{best_value:+.2f}" if best_value is not None else "N/A"
    success_rate = f"{100.0 * success_value:.1f}%" if success_value is not None else "N/A"
    collision_rate = f"{100.0 * collision_value:.1f}%" if collision_value is not None else "N/A"
    generation_worlds = attempt.generation_world_count or len(group)
    speed_text = f"{speed:.2f}" if speed is not None else "N/A"
    clearance_text = f"{clearance:.2f}" if clearance is not None else "N/A"
    min_clearance_text = (
        f"{attempt.min_clearance:.2f}" if attempt.min_clearance is not None else "N/A"
    )
    world_number = (
        attempt.world_index
        if attempt.world_index is not None
        else group.index(attempt) + 1
    )
    generation_text = str(attempt.generation) if attempt.generation is not None else "N/A"
    policy_text = f"v{attempt.policy_version}" if attempt.policy_version is not None else "N/A"
    panel = pygame.Rect(18, 625, 482, 112)
    translucent_panel(screen, panel)
    backend = (
        "DAYTONA"
        if attempt.daytona_verified
        else "LOCAL PREVIEW"
        if attempt.provenance == "LOCAL PREVIEW"
        else "RECORDED"
    )
    heading = (
        f"GEN {generation_text}  •  POLICY {policy_text}  •  "
        f"{generation_worlds} {backend} WORLDS"
    )
    screen.blit(text_font.render(heading, True, WHITE), (34, 636))
    if attempt.daytona_verified and attempt.next_policy_version is not None:
        trained = small_font.render(
            f"TRAINED v{attempt.next_policy_version}", True, GREEN
        )
        screen.blit(trained, (panel.right - trained.get_width() - 18, 640))

    row_one = (
        (34, f"AVG {average_reward}", CYAN),
        (164, f"BEST {best_reward}", GREEN),
        (298, f"SUCCESS/COLL {success_rate}/{collision_rate}", WHITE),
    )
    row_two = (
        (34, f"RWD {current_reward_text}/{reward_text}", WHITE),
        (190, f"SPEED {speed_text}", WHITE),
        (310, f"CLEAR {clearance_text}/{min_clearance_text}", WHITE),
    )
    for x, text, colour in row_one:
        screen.blit(small_font.render(text, True, colour), (x, 661))
    for x, text, colour in row_two:
        screen.blit(small_font.render(text, True, colour), (x, 681))

    identity = (
        f"WORLD {world_number}/{len(group)}  •  SEED "
        f"{attempt.seed if attempt.seed is not None else 'N/A'}  •  "
        f"POINT {cursor + 1}/{len(attempt.points)}  •  {status}"
    )
    screen.blit(small_font.render(identity, True, status_colour), (34, 701))
    sandbox_label = f"SANDBOX {sandbox_id}"
    screen.blit(
        small_font.render(
            sandbox_label,
            True,
            WHITE if attempt.sandbox_id else MUTED,
        ),
        (34, 719),
    )

    controls = pygame.Rect(18, HEIGHT - 61, 610, 43)
    translucent_panel(screen, controls)
    hint = "R  REPLAY THIS RUN     N  NEXT RECORDED UNIVERSE     ESC  EXIT"
    screen.blit(small_font.render(hint, True, MUTED), (34, HEIGHT - 47))


def draw_end_overlay(screen: Any, env: GravityEnv, fonts: tuple[Any, ...], elapsed: float) -> None:
    assert pygame is not None
    if not env.done:
        return

    title_font, text_font = fonts[:2]
    center = tuple(round(value) for value in _screen_point(_xy(env.ship_position)))
    effects = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    if env.success:
        effects.fill((22, 235, 171, 14))
        for index in range(5):
            radius = int(23 + index * 20 + (elapsed * 35) % 20)
            pygame.draw.circle(effects, (80, 255, 201, max(10, 90 - index * 15)), center, radius, 3)
        headline = "PORTAL REACHED"
        detail = "Success — press R to replay or N for a new universe"
        colour = GREEN
    else:
        effects.fill((255, 18, 55, 20))
        pulse = int(30 + 6 * math.sin(elapsed * 7))
        pygame.draw.circle(effects, (255, 57, 82, 110), center, pulse, 4)
        for index in range(12):
            angle = index * math.tau / 12
            inner = (
                center[0] + round(math.cos(angle) * 18),
                center[1] + round(math.sin(angle) * 18),
            )
            outer = (
                center[0] + round(math.cos(angle) * 52),
                center[1] + round(math.sin(angle) * 52),
            )
            pygame.draw.line(effects, (255, 108, 82, 140), inner, outer, 3)
        headline = "SHIP LOST"
        if "collision" in str(env.status):
            reason = "Collision"
        elif env.status == "out_of_bounds":
            reason = "Lost beyond the flight boundary"
        elif env.status == "timeout":
            reason = "Flight time exhausted"
        else:
            reason = str(env.status).replace("_", " ").title()
        detail = f"{reason} — press R to retry or N for a new universe"
        colour = RED

    viewport = pygame.Rect(
        WORLD_VIEWPORT_X,
        WORLD_VIEWPORT_Y,
        WORLD_VIEWPORT_WIDTH,
        WORLD_VIEWPORT_HEIGHT,
    )
    screen.blit(effects, viewport.topleft, viewport)
    viewport_center_x = 430
    banner = pygame.Rect(viewport_center_x - 285, HEIGHT // 2 - 62, 570, 124)
    translucent_panel(screen, banner)
    title = title_font.render(headline, True, colour)
    subtitle = text_font.render(detail, True, WHITE)
    screen.blit(title, title.get_rect(center=(viewport_center_x, HEIGHT // 2 - 19)))
    screen.blit(subtitle, subtitle.get_rect(center=(viewport_center_x, HEIGHT // 2 + 24)))


def draw_recorded_end_overlay(
    screen: Any,
    attempt: AttemptTrail,
    position: tuple[float, float],
    portal_position: tuple[float, float],
    fonts: tuple[Any, Any, Any],
    elapsed: float,
    *,
    finished: bool,
) -> None:
    """Make the recording's real terminal outcome impossible to miss."""

    assert pygame is not None
    if not finished:
        return
    title_font, text_font, _ = fonts
    screen_position = _screen_point(position)
    impact = (
        max(
            WORLD_VIEWPORT_X + 10,
            min(WORLD_VIEWPORT_X + WORLD_VIEWPORT_WIDTH - 10, round(screen_position[0])),
        ),
        max(
            WORLD_VIEWPORT_Y + 10,
            min(WORLD_VIEWPORT_Y + WORLD_VIEWPORT_HEIGHT - 10, round(screen_position[1])),
        ),
    )
    portal = tuple(round(value) for value in _screen_point(portal_position))
    effects = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    termination = (attempt.termination or "recorded_end").lower()

    if attempt.success is True:
        effects.fill((20, 255, 183, 25))
        pulse = int((elapsed * 62) % 38)
        for index in range(8):
            radius = 24 + index * 19 + pulse
            pygame.draw.circle(
                effects,
                (77, 255, 204, max(8, 138 - index * 15)),
                portal,
                radius,
                4,
            )
        pygame.draw.circle(effects, (220, 255, 255, 210), portal, 16)
        headline = "PORTAL ENTRY — SUCCESS"
        detail = "Recorded agent reached the target portal  •  R replay  •  N next universe"
        colour = GREEN
    elif "collision" in termination:
        effects.fill((255, 12, 39, 31))
        pulse = int(10 + (elapsed * 42) % 32)
        for radius, alpha in ((pulse + 14, 180), (pulse + 38, 115), (pulse + 72, 55)):
            pygame.draw.circle(effects, (255, 60, 78, alpha), impact, radius, 4)
        for index in range(20):
            angle = index * math.tau / 20
            inner = (
                impact[0] + round(math.cos(angle) * 15),
                impact[1] + round(math.sin(angle) * 15),
            )
            outer = (
                impact[0] + round(math.cos(angle) * (52 + pulse)),
                impact[1] + round(math.sin(angle) * (52 + pulse)),
            )
            pygame.draw.line(effects, (255, 128, 73, 185), inner, outer, 3)
        if "planet" in termination:
            headline = "PLANET IMPACT"
        elif "asteroid" in termination:
            headline = "ASTEROID IMPACT"
        else:
            headline = "COLLISION — SHIP LOST"
        detail = "Recorded failure returned from the selected universe  •  R replay  •  N next"
        colour = RED
    else:
        effects.fill((255, 124, 21, 19))
        pulse = int(30 + 8 * math.sin(elapsed * 6.0))
        pygame.draw.circle(effects, (255, 183, 66, 130), impact, pulse, 4)
        headline = termination.replace("_", " ").upper()
        detail = "Recorded run ended without portal entry  •  R replay  •  N next universe"
        colour = (255, 190, 74)

    viewport = pygame.Rect(
        WORLD_VIEWPORT_X,
        WORLD_VIEWPORT_Y,
        WORLD_VIEWPORT_WIDTH,
        WORLD_VIEWPORT_HEIGHT,
    )
    screen.blit(effects, viewport.topleft, viewport)
    viewport_center_x = 430
    banner_center_y = 540
    banner = pygame.Rect(viewport_center_x - 315, banner_center_y - 65, 630, 130)
    translucent_panel(screen, banner)
    title = title_font.render(headline, True, colour)
    subtitle = text_font.render(detail, True, WHITE)
    screen.blit(title, title.get_rect(center=(viewport_center_x, banner_center_y - 20)))
    screen.blit(subtitle, subtitle.get_rect(center=(viewport_center_x, banner_center_y + 26)))


def reset_world(env: GravityEnv, seed: int) -> None:
    env.reset(seed=seed)


def _completed_attempt(
    env: GravityEnv,
) -> AttemptTrail | None:
    points = tuple((float(point["x"]), float(point["y"])) for point in env.trajectory)
    if len(points) < 2:
        return None
    return AttemptTrail(
        points=points,
        reward=float(env.episode_reward),
        success=bool(env.success),
        seed=int(env.seed),
        trajectory=tuple(dict(point) for point in env.trajectory),
        termination=str(env.status) if env.done else None,
        min_clearance=float(env.min_clearance_seen),
        mean_speed=float(env.info()["mean_speed"]),
        max_speed=float(env.max_speed),
        provenance="LOCAL PREVIEW",
    )


def _source_label(attempts: list[AttemptTrail], current_seed: int) -> str:
    del attempts, current_seed
    return "LOCAL PREVIEW"


def _select_replay_group(
    attempts: list[AttemptTrail],
    *,
    generation: int | None,
    policy_version: int | None,
) -> list[AttemptTrail]:
    """Select one evaluated policy generation; never mix its summary metrics."""

    candidates = list(attempts)
    if generation is not None:
        candidates = [item for item in candidates if item.generation == generation]
        if not candidates:
            raise ValueError(f"no recorded rollout has generation {generation}")
    else:
        known_generations = [
            item.generation for item in candidates if item.generation is not None
        ]
        if known_generations:
            newest_generation = max(known_generations)
            candidates = [
                item for item in candidates if item.generation == newest_generation
            ]

    if policy_version is not None:
        candidates = [
            item for item in candidates if item.policy_version == policy_version
        ]
        if not candidates:
            suffix = f" in generation {generation}" if generation is not None else ""
            raise ValueError(
                f"no recorded rollout has policy version {policy_version}{suffix}"
            )
    else:
        known_policies = [
            item.policy_version for item in candidates if item.policy_version is not None
        ]
        if known_policies:
            newest_policy = max(known_policies)
            candidates = [
                item for item in candidates if item.policy_version == newest_policy
            ]

    if not candidates:
        raise ValueError("rollout artifact contains no displayable attempts")
    missing_seeds = [index for index, item in enumerate(candidates) if item.seed is None]
    if missing_seeds:
        raise ValueError(
            "every displayed rollout needs its universe seed; missing on index "
            + ", ".join(str(index) for index in missing_seeds)
        )
    return candidates


def _initial_replay_index(attempts: list[AttemptTrail], seed: int | None) -> int:
    if seed is not None:
        for index, attempt in enumerate(attempts):
            if attempt.seed == seed:
                return index
        raise ValueError(f"seed {seed} is not present in the selected rollout generation")
    for index, attempt in enumerate(attempts):
        if attempt.is_champion:
            return index
    scored = [
        (index, attempt)
        for index, attempt in enumerate(attempts)
        if attempt.reward is not None
    ]
    return (
        max(scored, key=lambda pair: float(pair[1].reward))[0]
        if scored
        else 0
    )


def _replay_environment(attempt: AttemptTrail) -> GravityEnv:
    """Rebuild and verify the recorded universe before drawing its trajectory."""

    if attempt.seed is None:
        raise ValueError("recorded replay is missing its universe seed")
    env = GravityEnv(seed=int(attempt.seed))
    if attempt.universe is None:
        raise ValueError(
            f"recorded replay for seed {attempt.seed} has no recorded universe"
        )
    if env.universe_dict() != attempt.universe:
        raise ValueError(
            f"recorded universe for seed {attempt.seed} does not match the current "
            "GravityEnv; rebuild and verify the Daytona snapshot before replay"
        )
    return env


def _validated_replay_environments(
    attempts: list[AttemptTrail],
) -> dict[int, GravityEnv]:
    """Verify every loaded universe, including candidates for ghost rendering."""

    environments: dict[int, GravityEnv] = {}
    for attempt in attempts:
        if attempt.seed is None:
            continue
        seed = int(attempt.seed)
        environments[seed] = _replay_environment(attempt)
    return environments


def _live_phase_description(state: LiveStateSnapshot | None) -> str:
    if state is None:
        return "READY"
    return state.phase.replace("_", " ")


def _draw_live_overlay(
    screen: Any,
    fonts: tuple[Any, Any, Any],
    *,
    state: LiveStateSnapshot | None,
    fallback_generation: int,
    fallback_policy_version: int,
    replay_mode: bool,
    locked: bool,
) -> None:
    """Draw the live controller state, lifecycle stream, and control guidance."""

    assert pygame is not None
    title_font, text_font, small_font = fonts
    generation = state.generation if state is not None else fallback_generation
    policy_version = (
        state.policy_version_used if state is not None else fallback_policy_version
    )
    phase = _live_phase_description(state)
    message = state.message if state is not None else "ready for input"

    outer = pygame.Rect(18, 18, 482, 192)
    translucent_panel(screen, outer)
    y = 31
    screen.blit(title_font.render("LIVE DAYTONA TRAINING", True, WHITE), (34, y))
    y += 31
    screen.blit(text_font.render(f"GENERATION {generation}", True, WHITE), (34, y))
    screen.blit(text_font.render(f"POLICY v{policy_version}", True, CYAN), (176, y))
    y += 24
    screen.blit(small_font.render(f"PHASE {phase}", True, GREEN), (34, y))

    current_status = small_font.render(
        f"STATUS {message[:58]}", True, MUTED
    )
    screen.blit(current_status, (34, y + 14))

    if state is None:
        worlds = ()
    else:
        worlds = state.worlds
    next_row = y + 34
    for world in worlds:
        lifecycle = world.lifecycle_state or "READY"
        sandbox_id = _short_identifier(world.sandbox_id or "no sandbox")
        reward_text = (
            f" {world.reward:+.2f}" if world.reward is not None else " n/a"
        )
        screen.blit(
            small_font.render(
                f"W{world.world_index:02d}: {lifecycle:<14}  {sandbox_id}  {reward_text}",
                True,
                WHITE,
            ),
            (34, next_row),
        )
        next_row += 16
        if next_row > 178:
            break

    controls_x = 34
    controls_y = outer.bottom - 17
    if locked:
        controls = "R LOCKED (training in progress)  •  ESC EXIT"
    elif replay_mode:
        controls = (
            "P REPLAY  •  N NEXT WORLD  •  R TRAIN NEXT GENERATION  •  "
            "ESC EXIT"
        )
    elif state is not None and state.phase in {"POLICY_UPDATED", "GENERATION_COMPLETE"}:
        controls = (
            "POLICY UPDATE COMPLETE  •  WEIGHTS CHANGED  •  "
            "PRESS R TO TRAIN NEXT GENERATION  •  ESC EXIT"
        )
    else:
        controls = "PRESS R TO TRAIN NEXT GENERATION  •  ESC EXIT"
    screen.blit(small_font.render(controls[:94], True, GREEN), (controls_x, controls_y))

    if state is not None and state.phase in {"POLICY_UPDATED", "GENERATION_COMPLETE"}:
        if state.policy_update is not None and state.policy_update.get("weights_changed") is True:
            screen.blit(
                small_font.render("REINFORCE UPDATE  •  WEIGHTS CHANGED", True, CYAN),
                (34, controls_y - 16),
            )
        if state.next_policy_version is not None:
            screen.blit(
                small_font.render(
                    f"POLICY v{state.policy_version_used} -> v{state.next_policy_version}",
                    True,
                    GREEN,
                ),
                (34, controls_y - 33),
            )


def run_live_demo(
    *args: Any,
    seed: int = 7,
    max_frames: int | None = None,
    worlds: int = DEFAULT_LIVE_WORLDS,
    max_steps: int = DEFAULT_LIVE_MAX_STEPS,
    runs_dir: Path = DEFAULT_LIVE_RUNS_DIR,
    checkpoint_dir: Path = DEFAULT_LIVE_CHECKPOINT_DIR,
    live_state_path: Path | None = None,
    snapshot_name: str = DEFAULT_LIVE_SNAPSHOT_NAME,
    start_generation: int = 0,
    start_policy_version: int = 0,
    start_checkpoint: Path | None = None,
) -> None:
    """Run the live interactive controller loop with live training status."""

    if len(args) != 0:
        raise TypeError("run_live_demo does not accept positional arguments")
    if pygame is None:
        raise SystemExit(
            "Pygame is required. Install the project dependencies, then run "
            "python visual_demo.py"
        )

    repo_root = Path.cwd().resolve()
    session_config = LiveTrainingConfig(
        repo_root=repo_root,
        snapshot_name=str(snapshot_name),
        worlds=int(worlds),
        max_steps=int(max_steps),
        runs_dir=Path(runs_dir),
        checkpoint_dir=Path(checkpoint_dir),
        live_state_path=(
            Path(live_state_path)
            if live_state_path is not None
            else repo_root / "runs" / LIVE_STATE_FILENAME
        ),
        base_seed=DEFAULT_LIVE_BASE_SEED,
        start_generation=int(start_generation),
        policy_version=int(start_policy_version),
        start_checkpoint=(
            Path(start_checkpoint) if start_checkpoint is not None else None
        ),
    )
    session = LiveTrainingSession(session_config)

    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()
        fonts = (
            pygame.font.SysFont("menlo", 24, bold=True),
            pygame.font.SysFont("menlo", 15),
            pygame.font.SysFont("menlo", 12, bold=True),
        )
        pygame.display.set_caption("Gravity Gauntlet — LIVE DAYTONA TRAINING")

        attempts: list[AttemptTrail] = []
        replay_attempts = attempts
        replay_mode = False
        replay_index = 0
        active_attempt = None
        current_seed = int(seed)
        preview_envs = {int(seed): GravityEnv(seed=seed)}
        env = preview_envs[current_seed]
        background, stars = make_background(current_seed)
        trail: deque[tuple[float, float]] = deque(
            [_xy(env.ship_position)], maxlen=TRAIL_LENGTH
        )
        ship_angle = -math.pi / 2
        replay_cursor = 0.0
        replay_end_frames = 0
        frame_count = 0
        running = True

        while running:
            session.poll(read_state=_read_live_state)
            completed = session.take_completed_generation()
            if completed is not None:
                attempts = completed
                replay_attempts = attempts
                try:
                    preview_envs = _validated_replay_environments(replay_attempts)
                except ValueError:
                    attempts = []
                    replay_attempts = []
                    replay_mode = False
                    active_attempt = None
                    env = preview_envs.get(current_seed, GravityEnv(seed=current_seed))
                    background, stars = make_background(current_seed)
                else:
                    replay_index = _initial_replay_index(
                        replay_attempts,
                        None,
                    )
                    active_attempt = replay_attempts[replay_index]
                    current_seed = int(active_attempt.seed)
                    env = preview_envs[current_seed]
                    background, stars = make_background(current_seed)
                    replay_cursor = 0.0
                    replay_end_frames = 0
                    ship_angle = -math.pi / 2
                    replay_mode = bool(replay_attempts)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        if replay_mode and replay_attempts:
                            replay_cursor = 0.0
                            replay_end_frames = 0
                            ship_angle = -math.pi / 2
                    elif event.key == pygame.K_n:
                        if replay_mode and len(replay_attempts) > 1:
                            replay_index = (replay_index + 1) % len(replay_attempts)
                            active_attempt = replay_attempts[replay_index]
                            current_seed = int(active_attempt.seed)
                            env = preview_envs[current_seed]
                            background, stars = make_background(current_seed)
                            replay_cursor = 0.0
                            replay_end_frames = 0
                            ship_angle = -math.pi / 2
                    elif event.key == pygame.K_r:
                        state = session.state
                        can_launch = (
                            not session.active
                            and (state is None or state.ready_for_next_generation)
                        )
                        if can_launch:
                            if session.request_generation():
                                replay_mode = False
                                attempts = []
                                replay_attempts = []
                                active_attempt = None
                                replay_cursor = 0.0
                                replay_end_frames = 0
                                ship_angle = -math.pi / 2
                                current_seed = int(seed)
                                env = GravityEnv(seed=current_seed)
                                preview_envs = {current_seed: env}
                                background, stars = make_background(current_seed)
                                trail = deque([_xy(env.ship_position)], maxlen=TRAIL_LENGTH)
                        else:
                            session.locked_notice = True

            if replay_mode and active_attempt is not None:
                assert active_attempt.seed is not None
                cursor = min(int(replay_cursor), len(active_attempt.points) - 1)
                position = active_attempt.points[cursor]
                recorded_velocity = _recorded_velocity(active_attempt, cursor)
                velocity = recorded_velocity if recorded_velocity is not None else (0.0, 0.0)
                thrust = _recorded_thrust(active_attempt, cursor)
            else:
                thrust = (0.0, 0.0)
                if env.done:
                    # READY-screen attract loop only: restart the same deterministic
                    # preview so the live UI never appears frozen between generations.
                    env = GravityEnv(seed=current_seed)
                    preview_envs[current_seed] = env
                    trail = deque([_xy(env.ship_position)], maxlen=TRAIL_LENGTH)
                    ship_angle = -math.pi / 2
                env.step(thrust)
                position = _xy(env.ship_position)
                velocity = _xy(env.ship_velocity)
                if math.hypot(*velocity) > 0.05:
                    ship_angle = math.atan2(velocity[1], velocity[0])
                if math.hypot(position[0] - trail[-1][0], position[1] - trail[-1][1]) >= 0.25:
                    trail.append(position)

            if replay_mode and active_attempt is not None:
                assert active_attempt.seed is not None
                assert active_attempt.world_index is not None
                if cursor < len(active_attempt.points) - 1:
                    replay_cursor = min(
                        len(active_attempt.points) - 1,
                        replay_cursor
                        + max(
                            0.05,
                            (len(active_attempt.points) - 1)
                            / (REPLAY_TARGET_SECONDS * FPS),
                        ),
                    )
                    replay_end_frames = 0
                else:
                    replay_end_frames += 1
                    if (
                        len(replay_attempts) > 1
                        and replay_end_frames >= round(REPLAY_END_HOLD_SECONDS * FPS)
                    ):
                        replay_index = (replay_index + 1) % len(replay_attempts)
                        active_attempt = replay_attempts[replay_index]
                        current_seed = int(active_attempt.seed)
                        env = preview_envs[current_seed]
                        background, stars = make_background(current_seed)
                        replay_cursor = 0.0
                        replay_end_frames = 0
                        ship_angle = -math.pi / 2
            elif math.hypot(*velocity) > 0.05:
                ship_angle = math.atan2(velocity[1], velocity[0])

            elapsed = pygame.time.get_ticks() / 1000.0
            screen.blit(background, (0, 0))
            draw_stars(screen, stars, elapsed)

            if replay_mode and active_attempt is not None:
                draw_recorded_trail(screen, active_attempt, cursor)
            else:
                draw_trail(screen, trail)

            draw_planets(screen, env.planets, current_seed)
            draw_asteroids(screen, env.asteroids, current_seed)
            draw_portal(screen, env.portal, elapsed)
            draw_speed_streaks(screen, position, velocity)
            draw_ship(screen, position, ship_angle, thrust, float(env.ship_radius))

            _draw_live_overlay(
                screen,
                fonts,
                state=session.state,
                fallback_generation=session.current_generation,
                fallback_policy_version=session.current_policy_version,
                replay_mode=bool(replay_attempts),
                locked=session.active,
            )

            if replay_mode and active_attempt is not None:
                draw_parallel_universes(
                    screen,
                    replay_attempts,
                    active_attempt,
                    preview_envs,
                    fonts,
                )
                draw_lifecycle_panel(screen, active_attempt, fonts)
                if attempts:
                    draw_learning_history_panel(screen, attempts, fonts)

            draw_recorded_end_overlay(
                screen,
                active_attempt or attempts[0],
                position,
                _position(env.portal),
                fonts,
                elapsed,
                finished=False,
            ) if replay_mode and active_attempt is not None else None

            pygame.display.flip()

            clock.tick(FPS)
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                running = False
    finally:
        pygame.quit()


def run_demo(
    seed: int | None = 7,
    max_frames: int | None = None,
    *,
    rollout_trails: list[AttemptTrail] | None = None,
    generation: int | None = None,
    policy_version: int | None = None,
) -> None:
    """Run the game; ``max_frames`` supports a bounded headless smoke test."""

    if pygame is None:
        raise SystemExit(
            "Pygame is required. Install the project dependencies, then run "
            "python visual_demo.py"
        )

    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        attempts = list(rollout_trails or [])
        replay_attempts = (
            _select_replay_group(
                attempts,
                generation=generation,
                policy_version=policy_version,
            )
            if attempts
            else []
        )
        replay_mode = bool(replay_attempts)
        pygame.display.set_caption(
            "Gravity Gauntlet — Recorded Daytona Experience"
            if replay_mode and all(item.daytona_verified for item in replay_attempts)
            else "Gravity Gauntlet — LOCAL PREVIEW"
            if replay_mode and all(item.provenance == "LOCAL PREVIEW" for item in replay_attempts)
            else "Gravity Gauntlet — Unverified Recorded Replay"
            if replay_mode
            else "Gravity Gauntlet — LOCAL PREVIEW"
        )
        clock = pygame.time.Clock()
        fonts = (
            pygame.font.SysFont("menlo", 24, bold=True),
            pygame.font.SysFont("menlo", 15),
            pygame.font.SysFont("menlo", 12, bold=True),
        )

        replay_index = _initial_replay_index(replay_attempts, seed) if replay_mode else 0
        active_attempt = replay_attempts[replay_index] if replay_mode else None
        current_seed = int(active_attempt.seed) if active_attempt is not None else int(seed or 7)
        validated_envs = _validated_replay_environments(attempts)
        preview_envs = {
            int(attempt.seed): validated_envs[int(attempt.seed)]
            for attempt in replay_attempts
            if attempt.seed is not None
        }
        env = preview_envs[current_seed] if replay_mode else GravityEnv(seed=current_seed)
        if not replay_mode:
            reset_world(env, current_seed)
        background, stars = make_background(current_seed)
        trail: deque[tuple[float, float]] = deque([_xy(env.ship_position)], maxlen=TRAIL_LENGTH)
        ship_angle = -math.pi / 2
        replay_cursor = 0.0
        replay_end_frames = 0
        frame_count = 0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        if replay_mode:
                            replay_cursor = 0.0
                            replay_end_frames = 0
                            ship_angle = -math.pi / 2
                        else:
                            completed = _completed_attempt(env)
                            if completed is not None:
                                attempts.append(completed)
                                attempts = attempts[-24:]
                            reset_world(env, current_seed)
                            trail = deque([_xy(env.ship_position)], maxlen=TRAIL_LENGTH)
                    elif event.key == pygame.K_n:
                        if replay_mode:
                            replay_index = (replay_index + 1) % len(replay_attempts)
                            active_attempt = replay_attempts[replay_index]
                            current_seed = int(active_attempt.seed)
                            env = preview_envs[current_seed]
                            background, stars = make_background(current_seed)
                            replay_cursor = 0.0
                            replay_end_frames = 0
                            ship_angle = -math.pi / 2
                        else:
                            completed = _completed_attempt(env)
                            if completed is not None:
                                attempts.append(completed)
                                attempts = attempts[-24:]
                            current_seed = (current_seed + 1) % 2_147_483_647
                            reset_world(env, current_seed)
                            background, stars = make_background(current_seed)
                            trail = deque([_xy(env.ship_position)], maxlen=TRAIL_LENGTH)

            if replay_mode:
                assert active_attempt is not None
                cursor = min(int(replay_cursor), len(active_attempt.points) - 1)
                position = active_attempt.points[cursor]
                recorded_velocity = _recorded_velocity(active_attempt, cursor)
                velocity = recorded_velocity if recorded_velocity is not None else (0.0, 0.0)
                thrust = _recorded_thrust(active_attempt, cursor)
            else:
                thrust = keyboard_action()
                if not env.done:
                    # GravityEnv is the single source of truth for every state change.
                    env.step(thrust)
                position = _xy(env.ship_position)
                velocity = _xy(env.ship_velocity)

            if math.hypot(*velocity) > 0.05:
                ship_angle = math.atan2(velocity[1], velocity[0])
            if not replay_mode and math.hypot(
                position[0] - trail[-1][0], position[1] - trail[-1][1]
            ) >= 0.25:
                trail.append(position)

            elapsed = pygame.time.get_ticks() / 1000.0
            screen.blit(background, (0, 0))
            draw_stars(screen, stars, elapsed)
            draw_ghost_trails(
                screen,
                attempts,
                current_seed,
                active_attempt if replay_mode else None,
            )
            if replay_mode:
                assert active_attempt is not None
                draw_recorded_trail(screen, active_attempt, cursor)
            else:
                draw_trail(screen, trail)
            draw_planets(screen, env.planets, current_seed)
            draw_asteroids(screen, env.asteroids, current_seed)
            draw_portal(screen, env.portal, elapsed)
            draw_speed_streaks(screen, position, velocity)
            draw_ship(screen, position, ship_angle, thrust, float(env.ship_radius))
            if replay_mode:
                assert active_attempt is not None
                finished = cursor >= len(active_attempt.points) - 1
                draw_replay_hud(screen, active_attempt, replay_attempts, cursor, fonts)
                draw_parallel_universes(
                    screen,
                    replay_attempts,
                    active_attempt,
                    preview_envs,
                    fonts,
                )
                draw_replay_banner(screen, active_attempt, fonts)
                draw_lifecycle_panel(screen, active_attempt, fonts)
                draw_learning_history_panel(screen, attempts, fonts)
                draw_recorded_end_overlay(
                    screen,
                    active_attempt,
                    position,
                    _position(env.portal),
                    fonts,
                    elapsed,
                    finished=finished,
                )
            else:
                draw_hud(
                    screen,
                    env,
                    current_seed,
                    fonts,
                    attempts,
                    generation=generation,
                    policy_version=policy_version,
                    source_label=_source_label(attempts, current_seed),
                )
                draw_end_overlay(screen, env, fonts, elapsed)
            pygame.display.flip()

            clock.tick(FPS)
            if replay_mode:
                assert active_attempt is not None
                if cursor < len(active_attempt.points) - 1:
                    replay_cursor = min(
                        len(active_attempt.points) - 1,
                        replay_cursor
                        + max(
                            0.05,
                            (len(active_attempt.points) - 1)
                            / (REPLAY_TARGET_SECONDS * FPS),
                        ),
                    )
                    replay_end_frames = 0
                else:
                    replay_end_frames += 1
                    if (
                        len(replay_attempts) > 1
                        and replay_end_frames >= round(REPLAY_END_HOLD_SECONDS * FPS)
                    ):
                        replay_index = (replay_index + 1) % len(replay_attempts)
                        active_attempt = replay_attempts[replay_index]
                        current_seed = int(active_attempt.seed)
                        env = preview_envs[current_seed]
                        background, stars = make_background(current_seed)
                        replay_cursor = 0.0
                        replay_end_frames = 0
                        ship_angle = -math.pi / 2
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                running = False
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live Daytona training or replay verified generation artifacts."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--rollouts",
        type=Path,
        help="worker/controller generation JSON to replay over seed-matched universes",
    )
    source.add_argument(
        "--local-preview",
        action="store_true",
        help="run the keyboard-driven local development mode; never labelled Daytona",
    )
    source.add_argument(
        "--live",
        action="store_true",
        help="run live Daytona integration training from the interactive panel",
    )
    parser.add_argument(
        "--live-worlds",
        type=int,
        default=DEFAULT_LIVE_WORLDS,
        help="number of parallel Daytona worlds for live mode",
    )
    parser.add_argument(
        "--live-max-steps",
        type=int,
        default=DEFAULT_LIVE_MAX_STEPS,
        help="maximum physics steps per live world",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="universe seed (default: loaded champion's seed, otherwise 7)",
    )
    parser.add_argument("--generation", type=int, help="real generation number to display")
    parser.add_argument("--policy-version", type=int, help="real policy version to display")
    args = parser.parse_args()
    if args.local_preview and (args.generation is not None or args.policy_version is not None):
        parser.error("--generation and --policy-version require --rollouts, not --local-preview")
    if args.live and (args.generation is not None or args.policy_version is not None or args.seed is not None):
        parser.error("--generation, --policy-version, and --seed are controlled by live replay artifacts")
    try:
        if args.rollouts:
            rollout_trails = load_rollout_trails(args.rollouts)
            selected_group = _select_replay_group(
                rollout_trails,
                generation=args.generation,
                policy_version=args.policy_version,
            )
            replay_index = _initial_replay_index(selected_group, args.seed)
            selected = selected_group[replay_index]
            seed = int(selected.seed)
            generation = selected.generation
            policy_version = selected.policy_version
        else:
            seed = args.seed if args.seed is not None else 7
            generation = args.generation
            policy_version = args.policy_version
            rollout_trails = []

        if args.live:
            run_live_demo(
                seed=seed,
                max_frames=int(smoke_frames) if smoke_frames else None,
                worlds=args.live_worlds,
                max_steps=args.live_max_steps,
            )
            return

        if args.local_preview:
            seed = args.seed if args.seed is not None else 7
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"cannot load rollout replay: {exc}")
    else:
        if args.rollouts:
            smoke_frames = os.environ.get("GRAVITY_DEMO_MAX_FRAMES")
            try:
                run_demo(
                    seed,
                    int(smoke_frames) if smoke_frames else None,
                    rollout_trails=rollout_trails,
                    generation=generation,
                    policy_version=policy_version,
                )
            except ValueError as exc:
                parser.error(f"cannot replay rollout: {exc}")
            return
        if args.local_preview:
            run_demo(
                seed,
                int(smoke_frames) if smoke_frames else None,
            )
            return

    smoke_frames = os.environ.get("GRAVITY_DEMO_MAX_FRAMES")
    smoke_frames = os.environ.get("GRAVITY_DEMO_MAX_FRAMES")
    if not (args.rollouts or args.local_preview or args.live):
        try:
            run_live_demo(
                seed=seed,
                max_frames=int(smoke_frames) if smoke_frames else None,
                worlds=args.live_worlds,
                max_steps=args.live_max_steps,
            )
        except ValueError as exc:
            parser.error(f"cannot run live loop: {exc}")


if __name__ == "__main__":
    main()
