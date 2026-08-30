"""End-to-end judge-demo coordinator for real Daytona generations.

This module owns orchestration only.  Physics remains in ``gravity_env.py`` /
``rollout_worker.py``, sandbox creation remains in ``daytona_orchestrator.py``,
and policy-gradient mathematics remains in ``trainer.py``.

There is intentionally no local rollout fallback.  A Daytona failure fails the
generation visibly and no policy update is attempted from partial results.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

from .demo_state import (
    GenerationState,
    LIVE_TRAINING_PHASES,
    LiveTrainingState,
    TrainingState,
    WorldState,
    json_safe,
    utc_timestamp,
    validate_visual_trajectory,
)


DEFAULT_WORLDS = 8
DEFAULT_GENERATIONS = 1
DEFAULT_MAX_STEPS = 500
DEFAULT_BASE_SEED = 18_473
DEFAULT_INITIAL_POLICY_VERSION = 0
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
LIVE_STATE_FILENAME = "live_state.json"
LIVE_PHASE_HISTORY_LIMIT = 32
TRAINER_METRIC_FIELDS = (
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
    "weights_changed",
    "mean_reward_to_go",
    "std_reward_to_go",
    "advantage_mean",
    "advantage_std",
)
REQUIRED_TRAINER_METRICS = (
    "episodes",
    "transitions",
    "loss",
    "entropy",
    "parameter_l2_delta",
    "changed_parameter_tensors",
    "changed_parameter_elements",
    "weights_changed",
)


class IntegrationUnavailableError(RuntimeError):
    """Raised when one of the real integration components is unavailable."""


class IntegrationContractError(RuntimeError):
    """Raised when real component output does not satisfy the shared contract."""


@dataclass(frozen=True, slots=True)
class IntegrationComponents:
    """The one canonical trainer entry point used by the glue layer."""

    run_daytona_training: Callable[..., Any]
    execution_backend: str = "fixture"


class LiveStateTracker:
    """Atomically publish genuine controller progress for a visual client."""

    def __init__(
        self,
        *,
        path: str | Path,
        invocation_id: str,
        execution_backend: str,
        generations: int,
        worlds: int,
        max_steps: int,
        start_generation: int,
        policy_version: int,
        latest_valid: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self._worlds: dict[int, dict[str, Any]] = {}
        self._collected_worlds: set[int] = set()
        self.state = LiveTrainingState(
            execution_backend=execution_backend,
            invocation_id=invocation_id,
            requested_generations=generations,
            requested_worlds=worlds,
            max_steps=max_steps,
            generation=start_generation,
            policy_version_used=policy_version,
            worlds_expected=worlds,
            latest_valid=(
                None if latest_valid is None else json_safe(dict(latest_valid))
            ),
        )
        self.transition(
            "READY",
            message=(
                f"generation {start_generation} / policy v{policy_version} ready"
            ),
            ready_for_next_generation=True,
        )

    def transition(self, phase: str, *, message: str, **changes: Any) -> None:
        if phase not in LIVE_TRAINING_PHASES:
            raise IntegrationContractError(f"unknown live-training phase: {phase}")
        for key, value in changes.items():
            if not hasattr(self.state, key):
                raise IntegrationContractError(
                    f"unknown live-training state field: {key}"
                )
            setattr(self.state, key, json_safe(value))
        self.state.phase = phase
        self.state.message = str(message)
        self.state.updated_at = utc_timestamp()
        self.state.revision += 1
        self.state.phase_history.append(
            {
                "revision": self.state.revision,
                "phase": phase,
                "generation": self.state.generation,
                "policy_version_used": self.state.policy_version_used,
                "next_policy_version": self.state.next_policy_version,
                "experiences_collected": self.state.experiences_collected,
                "timestamp": self.state.updated_at,
            }
        )
        if len(self.state.phase_history) > LIVE_PHASE_HISTORY_LIMIT:
            del self.state.phase_history[:-LIVE_PHASE_HISTORY_LIMIT]
        self._write()

    def prepare_generation(self, generation: int, policy_version: int) -> None:
        self._worlds = {
            world_index: {
                "world_index": world_index,
                "seed": None,
                "sandbox_id": None,
                "lifecycle_state": None,
                "reward": None,
                "success": None,
                "termination": None,
                "result_collected": False,
                "error": None,
                "lifecycle": [],
            }
            for world_index in range(1, self.state.worlds_expected + 1)
        }
        self._collected_worlds.clear()
        self.transition(
            "FREEZING_POLICY",
            message=f"freezing policy v{policy_version}",
            generation=int(generation),
            policy_version_used=int(policy_version),
            next_generation=None,
            next_policy_version=None,
            sandboxes_live=0,
            experiences_collected=0,
            worlds=self._ordered_worlds(),
            trainer_metrics=None,
            policy_update=None,
            generation_json=None,
            ready_for_next_generation=False,
            interrupted=False,
            error=None,
        )

    def lifecycle_event(self, event: Mapping[str, Any]) -> None:
        captured = json_safe(dict(event))
        generation = _non_negative_int(captured.get("generation"), "generation")
        policy_version = _non_negative_int(
            captured.get("policy_version"), "policy_version"
        )
        if (
            generation != self.state.generation
            or policy_version != self.state.policy_version_used
        ):
            raise IntegrationContractError(
                "live lifecycle identity does not match the active frozen policy: "
                f"generation={generation}, policy=v{policy_version}, expected "
                f"generation={self.state.generation}, "
                f"policy=v{self.state.policy_version_used}"
            )
        world_index = _positive_int(captured.get("world"), "world")
        if world_index > self.state.worlds_expected:
            raise IntegrationContractError(
                f"live lifecycle world {world_index} exceeds requested world count "
                f"{self.state.worlds_expected}"
            )
        lifecycle_state = str(captured.get("state", ""))
        if not lifecycle_state:
            raise IntegrationContractError("live lifecycle state is required")
        world = self._worlds[world_index]
        self._merge_identity(world, captured, field="seed")
        self._merge_identity(world, captured, field="sandbox_id")
        world["lifecycle_state"] = lifecycle_state
        if captured.get("reward") is not None:
            world["reward"] = captured["reward"]
        if captured.get("termination") is not None:
            world["termination"] = captured["termination"]
        if lifecycle_state == "SUCCESS":
            world["success"] = True
        elif lifecycle_state in {"COLLISION", "OUT_OF_BOUNDS", "TIMEOUT"}:
            world["success"] = False
        if lifecycle_state == "ERROR":
            world["error"] = captured.get("error") or "Daytona world error"
        if lifecycle_state == "RESULT_COLLECTED":
            world["result_collected"] = True
            self._collected_worlds.add(world_index)
        world["lifecycle"].append(captured)
        if len(world["lifecycle"]) > 16:
            del world["lifecycle"][:-16]

        self.state.worlds = self._ordered_worlds()
        self.state.sandboxes_live = sum(
            bool(item.get("sandbox_id")) for item in self._worlds.values()
        )
        self.state.experiences_collected = len(self._collected_worlds)
        if self._collected_worlds:
            phase = "COLLECTING_EXPERIENCES"
            message = (
                f"{len(self._collected_worlds)}/{self.state.worlds_expected} "
                "experiences collected"
            )
        elif lifecycle_state == "CREATING":
            phase = "CREATING_DAYTONA_WORLDS"
            message = "creating Daytona worlds"
        else:
            phase = "RUNNING_DAYTONA"
            message = (
                f"{self.state.sandboxes_live}/{self.state.worlds_expected} "
                "Daytona worlds live"
            )
        if self.state.phase != phase:
            self.transition(phase, message=message)
        else:
            self.state.message = message
            self._write_revision()

    def training_started(self) -> None:
        self.transition(
            "TRAINING_POLICY",
            message=(
                f"REINFORCE update from {self.state.experiences_collected}/"
                f"{self.state.worlds_expected} experiences"
            ),
        )

    def generation_completed(
        self,
        *,
        generation: int,
        policy_version: int,
        next_policy_version: int,
        record: Mapping[str, Any],
        policy_update: Mapping[str, Any],
        generation_path: str | Path,
    ) -> None:
        metrics = trainer_metrics_from_record(
            record,
            expected_episodes=self.state.worlds_expected,
        )
        if policy_update.get("weights_changed") is not True:
            raise IntegrationContractError(
                "controller checkpoint proof did not verify changed weights"
            )
        completed_path = str(Path(generation_path))
        latest_valid = {
            "generation": int(generation),
            "policy_version_used": int(policy_version),
            "next_policy_version": int(next_policy_version),
            "generation_json": completed_path,
            "checkpoint": policy_update["next_checkpoint"],
            "checkpoint_model_sha256": policy_update["next_model_sha256"],
        }
        self.transition(
            "POLICY_UPDATED",
            message=f"weights changed: policy v{policy_version} -> v{next_policy_version}",
            next_generation=int(generation) + 1,
            next_policy_version=int(next_policy_version),
            trainer_metrics=metrics,
            policy_update=json_safe(dict(policy_update)),
            generation_json=completed_path,
            latest_valid=latest_valid,
        )
        self.transition(
            "GENERATION_COMPLETE",
            message=(
                f"generation {generation} complete; policy v{next_policy_version} ready"
            ),
            completed_generations=self.state.completed_generations + 1,
            ready_for_next_generation=True,
        )

    def launch_next_generation(self, generation: int, policy_version: int) -> None:
        self.transition(
            "LAUNCHING_NEXT_GENERATION",
            message=(
                f"launching generation {generation} with policy v{policy_version}"
            ),
            next_generation=int(generation),
            next_policy_version=int(policy_version),
            ready_for_next_generation=False,
        )
        self.prepare_generation(generation, policy_version)

    def fail(self, error: BaseException, *, interrupted: bool = False) -> None:
        self.transition(
            "ERROR",
            message=("training interrupted" if interrupted else "training failed"),
            ready_for_next_generation=False,
            interrupted=bool(interrupted),
            error={
                "type": type(error).__name__,
                "message": str(error),
            },
        )

    def _merge_identity(
        self,
        world: dict[str, Any],
        event: Mapping[str, Any],
        *,
        field: str,
    ) -> None:
        incoming = event.get(field)
        if incoming is None:
            return
        current = world.get(field)
        if current is not None and current != incoming:
            raise IntegrationContractError(
                f"world {world['world_index']} emitted conflicting {field} values"
            )
        world[field] = incoming

    def _ordered_worlds(self) -> list[dict[str, Any]]:
        return [dict(self._worlds[index]) for index in sorted(self._worlds)]

    def _write_revision(self) -> None:
        self.state.updated_at = utc_timestamp()
        self.state.revision += 1
        self._write()

    def _write(self) -> None:
        _atomic_write_json(self.path, self.state.to_dict())


class LifecycleCollector:
    """Collect and display lifecycle events emitted by Agent 1."""

    def __init__(self, *, echo: bool = True) -> None:
        self.echo = bool(echo)
        self._events: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    def __call__(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping):
            raise IntegrationContractError("Daytona lifecycle event must be a mapping")
        try:
            world_index = int(event["world"])
            state = str(event["state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrationContractError(
                "Daytona lifecycle event requires world and state"
            ) from exc
        if world_index <= 0 or not state:
            raise IntegrationContractError(
                "Daytona lifecycle event has invalid world or state"
            )

        captured = json_safe(dict(event))
        generation = int(captured.get("generation", 0))
        self._events[(generation, world_index)].append(captured)
        if self.echo:
            sandbox_id = captured.get("sandbox_id") or "-"
            suffix = ""
            if captured.get("reward") is not None:
                suffix = f" reward={float(captured['reward']):.3f}"
            if captured.get("error"):
                suffix = f" error={captured['error']}"
            print(
                f"WORLD {world_index:02d} {state:<16} {sandbox_id}{suffix}",
                flush=True,
            )

    def events_for(
        self,
        world_index: int,
        *,
        generation: int = 0,
    ) -> list[dict[str, Any]]:
        return [
            dict(event)
            for event in self._events.get((int(generation), int(world_index)), [])
        ]

    def as_dict(self, *, generation: int = 0) -> dict[int, list[dict[str, Any]]]:
        return {
            world_index: [dict(event) for event in events]
            for (event_generation, world_index), events in self._events.items()
            if event_generation == int(generation)
        }


def load_integration_components() -> IntegrationComponents:
    """Load the canonical trainer entry point.

    The trainer owns policy creation, versioning, the call to
    ``daytona_orchestrator.run_generation``, and the policy-gradient update.
    This controller only persists and presents completed generation results.
    """

    try:
        from trainer import run_daytona_training
    except ImportError as exc:
        raise IntegrationUnavailableError(
            "trainer.run_daytona_training is required; the integration has no "
            "local rollout fallback"
        ) from exc

    if not callable(run_daytona_training):
        raise IntegrationUnavailableError(
            "trainer.run_daytona_training is not callable"
        )

    return IntegrationComponents(
        run_daytona_training=run_daytona_training,
        execution_backend="daytona",
    )


def trainer_metrics_from_record(
    record: Mapping[str, Any],
    *,
    expected_episodes: int,
) -> dict[str, Any]:
    """Copy and validate only metrics the canonical trainer actually returned."""

    missing = [key for key in REQUIRED_TRAINER_METRICS if key not in record]
    if missing:
        raise IntegrationContractError(
            "trainer record is missing live proof metric(s): " + ", ".join(missing)
        )
    episodes = _positive_int(record["episodes"], "episodes")
    if episodes != int(expected_episodes):
        raise IntegrationContractError(
            f"trainer metrics report {episodes} episodes for "
            f"{expected_episodes} collected worlds"
        )
    _positive_int(record["transitions"], "transitions")
    _finite_number(record["loss"], "loss")
    _finite_number(record["entropy"], "entropy")
    if _finite_number(record["parameter_l2_delta"], "parameter_l2_delta") <= 0.0:
        raise IntegrationContractError("parameter_l2_delta must prove a real update")
    _positive_int(record["changed_parameter_tensors"], "changed_parameter_tensors")
    _positive_int(record["changed_parameter_elements"], "changed_parameter_elements")
    if record["weights_changed"] is not True:
        raise IntegrationContractError(
            "trainer did not prove that the policy weights changed"
        )
    metrics = {
        key: json_safe(record[key])
        for key in TRAINER_METRIC_FIELDS
        if key in record
    }
    # These short UI labels are direct aliases of real trainer fields.
    metrics["changed_tensors"] = metrics["changed_parameter_tensors"]
    metrics["changed_elements"] = metrics["changed_parameter_elements"]
    return metrics


def validate_generation_seeds(
    seeds: Sequence[int], *, worlds: int
) -> list[int]:
    """Validate that every simultaneous universe has a distinct seed."""

    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise IntegrationContractError("generation seeds must be a sequence")
    seed_values = [int(seed) for seed in seeds]
    if len(seed_values) != int(worlds):
        raise IntegrationContractError(
            f"seed generator returned {len(seed_values)} seeds for {worlds} worlds"
        )
    if len(seed_values) != len(set(seed_values)):
        raise IntegrationContractError("each Daytona world requires a different seed")
    return seed_values


def coerce_generation_results(value: Any) -> list[Mapping[str, Any]]:
    """Accept Agent 1's result list, with small envelope tolerance."""

    if isinstance(value, Mapping):
        for key in ("results", "rollouts", "worlds"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise IntegrationContractError(
            "daytona_orchestrator.run_generation must return a result sequence"
        )
    results = list(value)
    if not results:
        raise IntegrationContractError("Daytona returned an empty generation")
    if any(not isinstance(result, Mapping) for result in results):
        raise IntegrationContractError("every Daytona world result must be a mapping")
    return results


def validate_training_rollout_contract(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    expected_obs_dim: int,
    expected_seeds: Sequence[int] | None = None,
    expected_policy_version: int | None = None,
) -> None:
    """Enforce worker, trainer, renderer, and provenance contracts before mutation."""

    if expected_seeds is not None and len(rollouts) != len(expected_seeds):
        raise IntegrationContractError(
            f"Daytona returned {len(rollouts)} worlds for "
            f"{len(expected_seeds)} requested seeds"
        )

    for index, rollout in enumerate(rollouts, start=1):
        missing = [
            field
            for field in (
                "sandbox_id",
                "seed",
                "policy_version",
                "reward",
                "success",
                "termination",
                "trajectory",
                "observations",
                "actions",
                "rewards",
                "policy_mode",
            )
            if field not in rollout
        ]
        if missing:
            raise IntegrationContractError(
                f"Daytona world {index} is missing trainer field(s): "
                + ", ".join(missing)
            )
        sandbox_id = rollout["sandbox_id"]
        if not isinstance(sandbox_id, str) or not sandbox_id.strip():
            raise IntegrationContractError(
                f"Daytona world {index} requires a verified sandbox_id"
            )
        actual_seed = _required_int(rollout["seed"], "seed")
        if expected_seeds is not None and actual_seed != int(expected_seeds[index - 1]):
            raise IntegrationContractError(
                f"Daytona world {index} returned seed {actual_seed}; "
                f"expected {int(expected_seeds[index - 1])}"
            )
        actual_policy = _non_negative_int(
            rollout["policy_version"], "policy_version"
        )
        if (
            expected_policy_version is not None
            and actual_policy != int(expected_policy_version)
        ):
            raise IntegrationContractError(
                f"Daytona world {index} returned policy v{actual_policy}; "
                f"expected v{int(expected_policy_version)}"
            )
        expected_mode = (
            "seeded_random_v0" if actual_policy == 0 else "neural_policy"
        )
        if rollout["policy_mode"] != expected_mode:
            raise IntegrationContractError(
                f"Daytona world {index} policy_mode must be {expected_mode!r} "
                f"for policy v{actual_policy}"
            )
        success = rollout["success"]
        if not isinstance(success, bool):
            raise IntegrationContractError(
                f"Daytona world {index} success must be a JSON boolean"
            )
        termination = rollout["termination"]
        valid_terminations = {
            "success",
            "planet_collision",
            "asteroid_collision",
            "out_of_bounds",
            "timeout",
        }
        if termination not in valid_terminations:
            raise IntegrationContractError(
                f"Daytona world {index} has unknown termination {termination!r}"
            )
        if success != (termination == "success"):
            raise IntegrationContractError(
                f"Daytona world {index} success disagrees with termination"
            )
        try:
            validate_visual_trajectory(
                rollout["trajectory"],
                label=f"Daytona world {index} trajectory",
            )
        except (TypeError, ValueError) as exc:
            raise IntegrationContractError(str(exc)) from exc
        observations = rollout["observations"]
        actions = rollout["actions"]
        rewards = rollout["rewards"]
        if any(
            isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
            for value in (observations, actions, rewards)
        ):
            raise IntegrationContractError(
                f"Daytona world {index} observations/actions/rewards must be sequences"
            )
        step_count = len(observations)
        if step_count == 0:
            raise IntegrationContractError(
                f"Daytona world {index} has no trainer transitions"
            )
        if len(actions) != step_count or len(rewards) != step_count:
            raise IntegrationContractError(
                f"Daytona world {index} trainer sequence lengths differ"
            )
        for step, observation in enumerate(observations):
            if isinstance(observation, (str, bytes)) or not isinstance(
                observation, Sequence
            ):
                raise IntegrationContractError(
                    f"Daytona world {index} observation {step} must be a sequence"
                )
            if len(observation) != int(expected_obs_dim):
                raise IntegrationContractError(
                    f"Daytona world {index} observation {step} has dimension "
                    f"{len(observation)}; expected {expected_obs_dim}"
                )
            for value in observation:
                _finite_number(value, "observation")
        for action in actions:
            if isinstance(action, bool) or not isinstance(action, int):
                raise IntegrationContractError(
                    f"Daytona world {index} actions must be categorical integers"
                )
            if not 0 <= action <= 8:
                raise IntegrationContractError(
                    f"Daytona world {index} action {action} is outside [0, 8]"
                )
        for reward in rewards:
            _finite_number(reward, "reward")
        total_reward = _finite_number(rollout["reward"], "reward")
        if not math.isclose(
            total_reward,
            math.fsum(float(reward) for reward in rewards),
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        ):
            raise IntegrationContractError(
                f"Daytona world {index} reward disagrees with per-step rewards"
            )


def world_state_from_result(
    result: Mapping[str, Any],
    *,
    world_index: int,
    expected_seed: int,
    expected_policy_version: int,
    lifecycle: Sequence[Mapping[str, Any]] = (),
    execution_backend: str | None = None,
) -> WorldState:
    """Normalize one real rollout without manufacturing missing evidence."""

    if not isinstance(result, Mapping):
        raise IntegrationContractError("Daytona world result must be a mapping")

    actual_seed = _optional_int(result.get("seed"), "seed")
    if actual_seed is not None and actual_seed != int(expected_seed):
        raise IntegrationContractError(
            f"world {world_index} returned seed {actual_seed}; "
            f"expected {expected_seed}"
        )
    actual_policy = _optional_int(result.get("policy_version"), "policy_version")
    if actual_policy is not None and actual_policy != int(expected_policy_version):
        raise IntegrationContractError(
            f"world {world_index} returned policy v{actual_policy}; "
            f"expected v{expected_policy_version}"
        )

    sandbox_id = result.get("sandbox_id")
    if sandbox_id is not None and (
        not isinstance(sandbox_id, str) or not sandbox_id.strip()
    ):
        raise IntegrationContractError(
            f"world {world_index} returned an invalid sandbox_id"
        )

    reward = _extract_reward(result)
    termination = _extract_termination(result)
    raw_success = result.get("success")
    if raw_success is not None and not isinstance(raw_success, bool):
        raise IntegrationContractError(
            f"world {world_index} success must be a JSON boolean"
        )
    success = raw_success if isinstance(raw_success, bool) else termination == "success"
    trajectory = result.get("trajectory", [])
    if trajectory is None:
        trajectory = []
    if not isinstance(trajectory, (list, tuple)):
        raise IntegrationContractError(
            f"world {world_index} trajectory must be a sequence"
        )
    if trajectory:
        try:
            trajectory = validate_visual_trajectory(
                trajectory,
                label=f"world {world_index} trajectory",
            )
        except (TypeError, ValueError) as exc:
            raise IntegrationContractError(str(exc)) from exc
    actions = result.get("actions", [])
    if actions is None:
        actions = []
    if not isinstance(actions, (list, tuple)):
        raise IntegrationContractError(f"world {world_index} actions must be a sequence")

    lifecycle_events = [json_safe(dict(event)) for event in lifecycle]
    lifecycle_sandbox_ids: set[str] = set()
    for event in lifecycle_events:
        event_world = _optional_int(event.get("world"), "lifecycle world")
        if event_world is not None and event_world != int(world_index):
            raise IntegrationContractError(
                f"world {world_index} received lifecycle data for world {event_world}"
            )
        event_seed = _optional_int(event.get("seed"), "lifecycle seed")
        if event_seed is not None and event_seed != int(expected_seed):
            raise IntegrationContractError(
                f"world {world_index} lifecycle seed {event_seed} does not match "
                f"{expected_seed}"
            )
        event_sandbox_id = event.get("sandbox_id")
        if event_sandbox_id is not None:
            if not isinstance(event_sandbox_id, str) or not event_sandbox_id.strip():
                raise IntegrationContractError(
                    f"world {world_index} lifecycle has an invalid sandbox_id"
                )
            lifecycle_sandbox_ids.add(event_sandbox_id)
    if len(lifecycle_sandbox_ids) > 1:
        raise IntegrationContractError(
            f"world {world_index} lifecycle contains conflicting sandbox IDs"
        )
    if lifecycle_sandbox_ids:
        lifecycle_sandbox_id = next(iter(lifecycle_sandbox_ids))
        if sandbox_id is not None and lifecycle_sandbox_id != sandbox_id:
            raise IntegrationContractError(
                f"world {world_index} result sandbox_id does not match its "
                "Daytona lifecycle"
            )
        if sandbox_id is None:
            sandbox_id = lifecycle_sandbox_id
    status = _world_status(result, termination, lifecycle_events)
    episode_length = _episode_length(result, trajectory)
    min_clearance = _nested_number(
        result,
        ("min_clearance", "minimum_clearance", "min_clearance_seen"),
    )
    mean_speed = _nested_number(result, ("mean_speed", "average_speed"))
    max_speed = _nested_number(result, ("max_speed",))
    fuel_used = _nested_number(result, ("fuel_used",))

    standardized = {
        "sandbox_id",
        "seed",
        "policy_version",
        "status",
        "reward",
        "total_reward",
        "success",
        "termination",
        "trajectory",
        "actions",
        "min_clearance",
        "minimum_clearance",
        "min_clearance_seen",
        "episode_length",
        "steps",
        "mean_speed",
        "average_speed",
        "max_speed",
        "fuel_used",
        "lifecycle",
        "events",
    }
    extra = {
        str(key): json_safe(value)
        for key, value in result.items()
        if key not in standardized
    }

    return WorldState(
        world_index=int(world_index),
        seed=int(expected_seed),
        policy_version=int(expected_policy_version),
        sandbox_id=sandbox_id,
        status=status,
        reward=reward,
        success=success,
        termination=termination,
        trajectory=json_safe(list(trajectory)),
        actions=json_safe(list(actions)),
        execution_backend=execution_backend,
        min_clearance=min_clearance,
        episode_length=episode_length,
        mean_speed=mean_speed,
        max_speed=max_speed,
        fuel_used=fuel_used,
        lifecycle=lifecycle_events,
        extra=extra,
    )


def build_generation_state(
    *,
    generation: int,
    policy_version: int,
    seeds: Sequence[int],
    results: Any,
    lifecycle_events: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
    next_policy_version: int | None = None,
    extra: Mapping[str, Any] | None = None,
    execution_backend: str | None = None,
) -> GenerationState:
    """Build metrics and select the champion from collected real rollouts."""

    seed_values = [int(seed) for seed in seeds]
    rollout_results = coerce_generation_results(results)
    if len(rollout_results) != len(seed_values):
        raise IntegrationContractError(
            f"Daytona returned {len(rollout_results)} worlds for "
            f"{len(seed_values)} requested seeds"
        )
    event_map = lifecycle_events or {}
    worlds = [
        world_state_from_result(
            result,
            world_index=index,
            expected_seed=seed,
            expected_policy_version=policy_version,
            lifecycle=event_map.get(index, ()),
            execution_backend=execution_backend,
        )
        for index, (seed, result) in enumerate(
            zip(seed_values, rollout_results), start=1
        )
    ]
    sandbox_ids = [world.sandbox_id for world in worlds if world.sandbox_id]
    if len(sandbox_ids) != len(set(sandbox_ids)):
        raise IntegrationContractError(
            "Daytona returned duplicate sandbox IDs for different worlds"
        )

    rewards = [world.reward for world in worlds if world.reward is not None]
    episode_lengths = [
        world.episode_length
        for world in worlds
        if world.episode_length is not None
    ]
    clearances = [
        world.min_clearance
        for world in worlds
        if world.min_clearance is not None
    ]
    mean_speeds = [world.mean_speed for world in worlds if world.mean_speed is not None]
    max_speeds = [world.max_speed for world in worlds if world.max_speed is not None]
    fuel_values = [world.fuel_used for world in worlds if world.fuel_used is not None]

    champion = max(
        (world for world in worlds if world.reward is not None),
        key=lambda world: float(world.reward),
        default=None,
    )
    world_count = len(worlds)
    generation_extra = dict(extra or {})
    if len(rewards) != world_count:
        generation_extra["worlds_missing_reward"] = world_count - len(rewards)
    missing_sandbox_ids = sum(world.sandbox_id is None for world in worlds)
    missing_trajectories = sum(not world.trajectory for world in worlds)
    if missing_sandbox_ids:
        generation_extra["worlds_missing_sandbox_id"] = missing_sandbox_ids
    if missing_trajectories:
        generation_extra["worlds_missing_trajectory"] = missing_trajectories
    generation_complete = (
        len(rewards) == world_count
        and missing_sandbox_ids == 0
        and missing_trajectories == 0
        and all(world.status != "ERROR" for world in worlds)
    )

    return GenerationState(
        generation=int(generation),
        policy_version=int(policy_version),
        worlds=worlds,
        average_reward=_mean(rewards),
        best_reward=max(rewards) if rewards else None,
        worst_reward=min(rewards) if rewards else None,
        success_rate=sum(world.success for world in worlds) / world_count,
        collision_rate=sum(_is_collision(world) for world in worlds) / world_count,
        average_episode_length=_mean(episode_lengths),
        best_world=champion.world_index if champion is not None else None,
        best_sandbox_id=champion.sandbox_id if champion is not None else None,
        average_min_clearance=_mean(clearances),
        mean_speed=_mean(mean_speeds),
        max_speed=max(max_speeds) if max_speeds else None,
        fuel_used=sum(fuel_values) if fuel_values else None,
        next_policy_version=next_policy_version,
        status="COMPLETE" if generation_complete else "INCOMPLETE",
        extra=json_safe(generation_extra),
    )


def save_generation_json(
    generation_state: GenerationState, runs_dir: str | Path = DEFAULT_RUNS_DIR
) -> Path:
    """Persist one renderer-ready generation as strict JSON."""

    output_path = Path(runs_dir) / f"generation_{generation_state.generation:03d}.json"
    _atomic_write_json(output_path, generation_state.to_dict())
    return output_path


def save_training_state_json(
    training_state: TrainingState, runs_dir: str | Path = DEFAULT_RUNS_DIR
) -> Path:
    """Persist recent generations for ghost-trajectory rendering."""

    output_path = Path(runs_dir) / "training_state.json"
    _atomic_write_json(output_path, training_state.to_dict())
    return output_path


def checkpoint_model_digest(
    path: str | Path,
    *,
    expected_policy_version: int,
    expected_obs_dim: int | None = None,
) -> str:
    """Hash only the learned model tensors in one trainer checkpoint.

    File hashes are insufficient proof of learning because version metadata or
    optimizer state can change while policy weights remain identical.  This
    helper loads the trainer's safe checkpoint payload and hashes tensor names,
    dtypes, shapes, and bytes from ``model_state_dict`` only.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise IntegrationContractError(
            f"policy checkpoint does not exist: {checkpoint_path}"
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - trainer already requires torch
        raise IntegrationUnavailableError(
            "PyTorch is required to verify policy checkpoint weights"
        ) from exc

    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise IntegrationContractError(
            f"could not read policy checkpoint {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise IntegrationContractError(
            f"policy checkpoint {checkpoint_path} must contain a mapping"
        )
    version = payload.get("policy_version")
    if isinstance(version, bool) or version != expected_policy_version:
        raise IntegrationContractError(
            f"policy checkpoint {checkpoint_path} has version {version!r}; "
            f"expected {expected_policy_version}"
        )
    if expected_obs_dim is not None:
        checkpoint_obs_dim = payload.get("obs_dim")
        if (
            isinstance(checkpoint_obs_dim, bool)
            or checkpoint_obs_dim != int(expected_obs_dim)
        ):
            raise IntegrationContractError(
                f"policy checkpoint {checkpoint_path} has obs_dim "
                f"{checkpoint_obs_dim!r}; expected {expected_obs_dim}"
            )
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping) or not model_state:
        raise IntegrationContractError(
            f"policy checkpoint {checkpoint_path} has no model_state_dict"
        )

    digest = hashlib.sha256()
    for name in sorted(model_state, key=str):
        tensor = model_state[name]
        if not torch.is_tensor(tensor):
            raise IntegrationContractError(
                f"policy checkpoint tensor {name!r} is not a tensor"
            )
        # Clone into an exact contiguous CPU storage so raw bytes can be
        # hashed without NumPy (the project's lean virtualenv does not require
        # it).
        value = tensor.detach().cpu().contiguous().clone()
        descriptor = json.dumps(
            {
                "name": str(name),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        raw = bytes(value.untyped_storage())
        expected_bytes = value.numel() * value.element_size()
        if len(raw) != expected_bytes:
            raise IntegrationContractError(
                f"policy checkpoint tensor {name!r} has unexpected storage size"
            )
        digest.update(raw)
    return digest.hexdigest()


def verify_policy_checkpoint_update(
    *,
    checkpoint_dir: str | Path,
    policy_version: int,
    next_policy_version: int,
    trainer_checkpoint: Any,
    input_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Return JSON proof that vN and vN+1 contain different model weights."""

    checkpoint_root = Path(checkpoint_dir)
    input_path = (
        checkpoint_root / f"policy_v{policy_version:03d}.pt"
        if input_checkpoint is None
        else Path(input_checkpoint)
    )
    expected_next_path = checkpoint_root / f"policy_v{next_policy_version:03d}.pt"
    if not isinstance(trainer_checkpoint, (str, os.PathLike)):
        raise IntegrationContractError(
            "trainer generation record has no checkpoint path"
        )
    reported_next_path = Path(trainer_checkpoint)
    if reported_next_path.resolve() != expected_next_path.resolve():
        raise IntegrationContractError(
            "trainer checkpoint path does not match the next policy version: "
            f"{reported_next_path} != {expected_next_path}"
        )

    input_digest = checkpoint_model_digest(
        input_path,
        expected_policy_version=policy_version,
    )
    next_digest = checkpoint_model_digest(
        expected_next_path,
        expected_policy_version=next_policy_version,
    )
    if input_digest == next_digest:
        raise IntegrationContractError(
            f"REINFORCE produced unchanged model weights for policy "
            f"v{policy_version} -> v{next_policy_version}; refusing to rename "
            "an unchanged policy"
        )
    return {
        "input_checkpoint": str(input_path),
        "next_checkpoint": str(expected_next_path),
        "input_model_sha256": input_digest,
        "next_model_sha256": next_digest,
        "weights_changed": True,
    }


async def run_training_demo(
    *,
    generations: int = DEFAULT_GENERATIONS,
    worlds: int = DEFAULT_WORLDS,
    max_steps: int = DEFAULT_MAX_STEPS,
    base_seed: int = DEFAULT_BASE_SEED,
    start_generation: int = 0,
    initial_policy_version: int = DEFAULT_INITIAL_POLICY_VERSION,
    obs_dim: int | None = None,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    policy_checkpoint: str | Path | None = None,
    live_state_path: str | Path | None = None,
    invocation_id: str | None = None,
    snapshot_name: str | None = None,
    keep_sandboxes: bool = False,
    components: IntegrationComponents | None = None,
    echo_lifecycle: bool = True,
    allow_overwrite: bool = False,
) -> TrainingState:
    """Persist and present generations produced by the canonical trainer loop."""

    generations = _positive_int(generations, "generations")
    worlds = _positive_int(worlds, "worlds")
    max_steps = _positive_int(max_steps, "max_steps")
    start_generation = _non_negative_int(start_generation, "start_generation")
    initial_policy_version = _non_negative_int(
        initial_policy_version, "initial_policy_version"
    )
    if start_generation != initial_policy_version:
        raise IntegrationContractError(
            "start_generation must match initial_policy_version for one policy per "
            "generation"
        )
    if start_generation == 0 and policy_checkpoint is not None:
        raise IntegrationContractError(
            "start_generation 0 requires no policy_checkpoint; provide a checkpoint "
            "only when resuming v1+"
        )
    if start_generation > 0 and policy_checkpoint is None:
        raise IntegrationContractError(
            "resuming from generation 1+ requires policy_checkpoint"
        )
    observation_dimension = _resolve_observation_dim(obs_dim)
    if components is None:
        try:
            os.environ["DAYTONA_API_KEY"]
        except KeyError as exc:
            raise IntegrationUnavailableError(
                "DAYTONA_API_KEY is required; no checkpoint or local rollout was "
                "created because local fallback is disabled"
            ) from exc
    loaded = components or load_integration_components()
    execution_backend = loaded.execution_backend
    if not isinstance(execution_backend, str) or not execution_backend.strip():
        raise IntegrationContractError(
            "integration component must declare a non-empty execution backend"
        )
    execution_backend = execution_backend.strip().lower()
    if components is None and execution_backend != "daytona":
        raise IntegrationContractError(
            "the production controller must use the Daytona execution backend"
        )

    runs_path = Path(runs_dir)
    checkpoint_path = Path(checkpoint_dir)
    _preflight_output_paths(
        runs_dir=runs_path,
        checkpoint_dir=checkpoint_path,
        generations=generations,
        start_generation=start_generation,
        initial_policy_version=initial_policy_version,
        allow_overwrite=bool(allow_overwrite),
    )
    controller_invocation_id = (
        uuid.uuid4().hex if invocation_id is None else str(invocation_id).strip()
    )
    if not controller_invocation_id:
        raise IntegrationContractError("invocation_id must be a non-empty string")
    resolved_live_state_path = (
        runs_path / LIVE_STATE_FILENAME
        if live_state_path is None
        else Path(live_state_path)
    )
    live_state = LiveStateTracker(
        path=resolved_live_state_path,
        invocation_id=controller_invocation_id,
        execution_backend=execution_backend,
        generations=generations,
        worlds=worlds,
        max_steps=max_steps,
        start_generation=start_generation,
        policy_version=initial_policy_version,
    )
    training_state = TrainingState(
        current_generation=start_generation - 1,
        current_policy_version=initial_policy_version,
    )
    lifecycle = LifecycleCollector(echo=echo_lifecycle)
    validated: dict[int, tuple[list[int], list[Mapping[str, Any]]]] = {}
    persisted_identities: list[dict[str, Any]] = []

    def collect_lifecycle(event: Mapping[str, Any]) -> None:
        lifecycle(event)
        live_state.lifecycle_event(event)

    def validate_before_update(
        policy_version: int,
        seeds: Sequence[int],
        rollouts: Sequence[Mapping[str, Any]],
    ) -> None:
        seed_values = validate_generation_seeds(seeds, worlds=worlds)
        rollout_values = coerce_generation_results(rollouts)
        validate_training_rollout_contract(
            rollout_values,
            expected_obs_dim=observation_dimension,
            expected_seeds=seed_values,
            expected_policy_version=policy_version,
        )
        version = int(policy_version)
        if version in validated:
            raise IntegrationContractError(
                f"trainer validated policy v{version} more than once"
            )
        validated[version] = (seed_values, rollout_values)
        live_state.training_started()

    def persist_after_update(
        record: Mapping[str, Any],
        rollouts: Sequence[Mapping[str, Any]],
    ) -> None:
        generation = _non_negative_int(record.get("generation"), "generation")
        policy_version = _non_negative_int(
            record.get("policy_version"), "policy_version"
        )
        next_policy_version = _positive_int(
            record.get("next_policy_version"), "next_policy_version"
        )
        expected_generation = start_generation + len(persisted_identities)
        expected_policy_version = expected_generation
        if generation != expected_generation or policy_version != expected_policy_version:
            raise IntegrationContractError(
                "trainer generation identity is out of order: "
                f"generation={generation}, policy_version={policy_version}, "
                f"expected={expected_generation}"
            )
        if next_policy_version != policy_version + 1:
            raise IntegrationContractError(
                "trainer policy version must advance exactly once per generation"
            )
        if policy_version not in validated:
            raise IntegrationContractError(
                f"trainer persisted policy v{policy_version} without validated rollouts"
            )
        seeds, validated_rollouts = validated.pop(policy_version)
        recorded_seeds = validate_generation_seeds(
            record.get("seeds"),
            worlds=worlds,
        )
        if recorded_seeds != seeds:
            raise IntegrationContractError(
                "trainer generation record seeds do not match the validated "
                "Daytona rollout seed batch"
            )
        if list(rollouts) != validated_rollouts:
            raise IntegrationContractError(
                "trainer changed rollout structures between validation and persistence"
            )

        policy_update = verify_policy_checkpoint_update(
            checkpoint_dir=checkpoint_dir,
            policy_version=policy_version,
            next_policy_version=next_policy_version,
            trainer_checkpoint=record.get("checkpoint"),
            input_checkpoint=(
                policy_checkpoint
                if policy_version == start_generation and start_generation > 0
                else None
            ),
        )
        trainer_metrics_from_record(record, expected_episodes=worlds)

        training_fields = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "generation",
                "policy_version",
                "next_policy_version",
                "seeds",
                "checkpoint",
            }
        }
        generation_state = build_generation_state(
            generation=generation,
            policy_version=policy_version,
            seeds=seeds,
            results=validated_rollouts,
            lifecycle_events=lifecycle.as_dict(generation=generation),
            next_policy_version=next_policy_version,
            execution_backend=execution_backend,
            extra={
                "execution_backend": execution_backend,
                "seed_batch": seeds,
                "trainer_checkpoint": record.get("checkpoint"),
                "training": json_safe(training_fields),
                "policy_update": policy_update,
            },
        )
        incomplete_worlds = [
            world.world_index
            for world in generation_state.worlds
            if world.execution_backend != execution_backend
            or world.sandbox_id is None
            or world.reward is None
            or len(world.trajectory) < 2
            or not world.actions
            or world.termination is None
            or world.status == "ERROR"
            or not any(
                event.get("state") == "RESULT_COLLECTED"
                for event in world.lifecycle
            )
        ]
        if generation_state.status != "COMPLETE" or incomplete_worlds:
            raise IntegrationContractError(
                f"{execution_backend} generation did not provide complete "
                "sandbox/lifecycle/"
                "trajectory/action proof for every world; the trained generation "
                f"was not persisted (incomplete worlds: {incomplete_worlds})"
            )

        generation_path = save_generation_json(generation_state, runs_dir)
        training_state.add_generation(generation_state)
        save_training_state_json(training_state, runs_dir)
        persisted_identities.append(
            {
                "generation": generation,
                "policy_version": policy_version,
                "next_policy_version": next_policy_version,
                "seeds": list(seeds),
            }
        )
        live_state.generation_completed(
            generation=generation,
            policy_version=policy_version,
            next_policy_version=next_policy_version,
            record=record,
            policy_update=policy_update,
            generation_path=generation_path,
        )
        _print_generation_result(generation_state, generation_path)
        if len(persisted_identities) < generations:
            live_state.launch_next_generation(
                generation + 1,
                next_policy_version,
            )

    live_state.prepare_generation(start_generation, initial_policy_version)
    try:
        history = loaded.run_daytona_training(
            generations=generations,
            worlds=worlds,
            max_steps=max_steps,
            obs_dim=observation_dimension,
            base_seed=int(base_seed),
            checkpoint_dir=checkpoint_dir,
            start_generation=start_generation,
            initial_policy_version=initial_policy_version,
            policy_checkpoint=policy_checkpoint,
            snapshot_name=snapshot_name,
            keep_sandboxes=bool(keep_sandboxes),
            event_callback=collect_lifecycle,
            rollout_validator=validate_before_update,
            on_generation=persist_after_update,
        )
        if inspect.isawaitable(history):
            history = await history
    except asyncio.CancelledError as exc:
        live_state.fail(exc, interrupted=True)
        raise
    except IntegrationContractError as exc:
        live_state.fail(exc)
        raise
    except Exception as exc:
        wrapped = IntegrationContractError(
            "canonical Daytona training failed; no local rollout fallback was "
            f"used: {type(exc).__name__}: {exc}"
        )
        live_state.fail(wrapped)
        raise wrapped from exc

    try:
        if (
            isinstance(history, (str, bytes, Mapping))
            or not isinstance(history, Sequence)
            or len(history) != generations
        ):
            raise IntegrationContractError(
                "trainer did not return one completed record per requested generation"
            )
        if len(persisted_identities) != generations:
            raise IntegrationContractError(
                "trainer completed without persisting every validated generation"
            )
        if validated:
            raise IntegrationContractError(
                "trainer completed with validated rollout batches that were never persisted"
            )
        for index, (returned, persisted) in enumerate(
            zip(history, persisted_identities, strict=True)
        ):
            if not isinstance(returned, Mapping):
                raise IntegrationContractError(
                    f"trainer history record {index} is not a mapping"
                )
            returned_identity = {
                "generation": _non_negative_int(
                    returned.get("generation"), "generation"
                ),
                "policy_version": _non_negative_int(
                    returned.get("policy_version"), "policy_version"
                ),
                "next_policy_version": _positive_int(
                    returned.get("next_policy_version"), "next_policy_version"
                ),
                "seeds": validate_generation_seeds(
                    returned.get("seeds"),
                    worlds=worlds,
                ),
            }
            if returned_identity != persisted:
                raise IntegrationContractError(
                    f"trainer history record {index} does not match its persisted "
                    "generation identity"
                )
    except IntegrationContractError as exc:
        live_state.fail(exc)
        raise
    return training_state


def _print_generation_result(state: GenerationState, path: Path) -> None:
    def metric(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    print()
    print(f"Average reward: {metric(state.average_reward)}")
    print(f"Best reward: {metric(state.best_reward)}")
    print(f"Worst reward: {metric(state.worst_reward)}")
    print(f"Success rate: {state.success_rate:.1%}")
    print(f"Collision rate: {state.collision_rate:.1%}")
    print(f"Champion sandbox: {state.best_sandbox_id or 'n/a'}")
    champion = state.champion
    print(f"Champion seed: {champion.seed if champion is not None else 'n/a'}")
    print("\nPolicy updated:")
    print(f"v{state.policy_version} -> v{state.next_policy_version}")
    policy_update = state.extra.get("policy_update")
    weights_changed = (
        isinstance(policy_update, Mapping)
        and policy_update.get("weights_changed") is True
    )
    print(f"Model weights changed: {'yes' if weights_changed else 'unverified'}")
    print("\nSaved:")
    print(path)


def _extract_reward(result: Mapping[str, Any]) -> float | None:
    for key in ("reward", "total_reward", "episode_reward"):
        if result.get(key) is not None:
            return _finite_number(result[key], key)
    rewards = result.get("rewards")
    if isinstance(rewards, Sequence) and not isinstance(rewards, (str, bytes)):
        return sum(_finite_number(value, "reward") for value in rewards)
    transitions = result.get("transitions")
    if isinstance(transitions, Sequence) and not isinstance(
        transitions, (str, bytes)
    ):
        values = [
            item.get("reward")
            for item in transitions
            if isinstance(item, Mapping) and item.get("reward") is not None
        ]
        if values:
            return sum(_finite_number(value, "reward") for value in values)
    return _nested_number(result, ("episode_reward",))


def _extract_termination(result: Mapping[str, Any]) -> str | None:
    for key in ("termination", "terminal_reason"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    for container_name in ("info", "final_info"):
        container = result.get(container_name)
        if isinstance(container, Mapping):
            value = container.get("termination", container.get("status"))
            if isinstance(value, str) and value:
                return value
    return None


def _world_status(
    result: Mapping[str, Any],
    termination: str | None,
    lifecycle: Sequence[Mapping[str, Any]],
) -> str:
    terminal_states = {"SUCCESS", "COLLISION", "OUT_OF_BOUNDS", "TIMEOUT", "ERROR"}
    for event in reversed(lifecycle):
        state = event.get("state")
        if isinstance(state, str) and state.upper() in terminal_states:
            return state.upper()
    raw_status = result.get("status")
    if isinstance(raw_status, str) and raw_status.upper() in {
        "CREATING",
        "LIVE",
        "RUNNING",
        "SUCCESS",
        "COLLISION",
        "OUT_OF_BOUNDS",
        "TIMEOUT",
        "ERROR",
        "RESULT_COLLECTED",
    }:
        return raw_status.upper()
    if termination:
        lowered = termination.lower()
        if "collision" in lowered:
            return "COLLISION"
        if lowered == "success":
            return "SUCCESS"
        if lowered == "out_of_bounds":
            return "OUT_OF_BOUNDS"
        if lowered == "timeout":
            return "TIMEOUT"
    return "RESULT_COLLECTED"


def _episode_length(result: Mapping[str, Any], trajectory: Sequence[Any]) -> int | None:
    for key in ("steps", "episode_length"):
        if result.get(key) is not None:
            value = _required_int(result[key], key)
            if value < 0:
                raise IntegrationContractError(f"{key} cannot be negative")
            return value
    for key in ("actions", "rewards", "observations", "transitions"):
        value = result.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return len(value)
    return max(0, len(trajectory) - 1) if trajectory else None


def _nested_number(result: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    containers: list[Mapping[str, Any]] = [result]
    for name in ("info", "final_info", "metrics"):
        candidate = result.get(name)
        if isinstance(candidate, Mapping):
            containers.append(candidate)
    for container in containers:
        for key in keys:
            value = container.get(key)
            if value is not None:
                return _finite_number(value, key)
    return None


def _is_collision(world: WorldState) -> bool:
    return bool(
        (world.termination and "collision" in world.termination.lower())
        or world.status == "COLLISION"
        or world.extra.get("collision")
    )


def _mean(values: Sequence[float | int]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrationContractError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise IntegrationContractError(f"{name} must be finite")
    return number


def _required_int(value: Any, name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise IntegrationContractError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise IntegrationContractError(f"{name} must be an integer")
    return integer


def _optional_int(value: Any, name: str) -> int | None:
    return None if value is None else _required_int(value, name)


def _positive_int(value: Any, name: str) -> int:
    integer = _required_int(value, name)
    if integer <= 0:
        raise IntegrationContractError(f"{name} must be positive")
    return integer


def _non_negative_int(value: Any, name: str) -> int:
    integer = _required_int(value, name)
    if integer < 0:
        raise IntegrationContractError(f"{name} cannot be negative")
    return integer


def _resolve_observation_dim(obs_dim: int | None) -> int:
    if obs_dim is not None:
        return _positive_int(obs_dim, "obs_dim")
    try:
        from gravity_env import OBSERVATION_DIM
    except (ImportError, AttributeError) as exc:
        raise IntegrationUnavailableError(
            "gravity_env.OBSERVATION_DIM is required; pass --obs-dim only for "
            "an explicitly compatible worker snapshot"
        ) from exc
    return _positive_int(OBSERVATION_DIM, "obs_dim")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        json_safe(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _preflight_output_paths(
    *,
    runs_dir: Path,
    checkpoint_dir: Path,
    generations: int,
    start_generation: int,
    initial_policy_version: int,
    allow_overwrite: bool,
) -> None:
    if allow_overwrite:
        return
    start_generation = _non_negative_int(start_generation, "start_generation")
    initial_policy_version = _non_negative_int(
        initial_policy_version,
        "initial_policy_version",
    )
    targets = [
        runs_dir / f"generation_{generation:03d}.json"
        for generation in range(start_generation, start_generation + generations)
    ]
    if start_generation == 0:
        targets.append(runs_dir / "training_state.json")
    start_checkpoint = 0 if start_generation == 0 else initial_policy_version + 1
    end_checkpoint = initial_policy_version + generations
    targets.extend(
        checkpoint_dir / f"policy_v{version:03d}.pt"
        for version in range(start_checkpoint, end_checkpoint + 1)
    )
    existing = [path for path in targets if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise IntegrationContractError(
            "refusing to overwrite generated demo artifacts: "
            f"{rendered}; choose new output directories or pass --overwrite"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real Daytona worlds, update the policy, and save judge-demo state."
    )
    parser.add_argument("--worlds", type=int, default=DEFAULT_WORLDS)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--obs-dim", type=int)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--start-generation", type=int, default=0)
    parser.add_argument(
        "--initial-policy-version",
        type=int,
        default=DEFAULT_INITIAL_POLICY_VERSION,
    )
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument(
        "--live-state-path",
        type=Path,
        help="atomic visual bridge (default: <runs-dir>/live_state.json)",
    )
    parser.add_argument(
        "--invocation-id",
        help="optional visual/controller correlation token; never a sandbox ID",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR
    )
    parser.add_argument("--snapshot-name")
    parser.add_argument("--keep-sandboxes", action="store_true")
    parser.add_argument("--quiet-lifecycle", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing generated run/checkpoint files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        asyncio.run(
            run_training_demo(
                generations=args.generations,
                worlds=args.worlds,
                max_steps=args.max_steps,
                base_seed=args.base_seed,
                obs_dim=args.obs_dim,
                runs_dir=args.runs_dir,
                checkpoint_dir=args.checkpoint_dir,
                start_generation=args.start_generation,
                initial_policy_version=args.initial_policy_version,
                policy_checkpoint=args.policy_checkpoint,
                live_state_path=args.live_state_path,
                invocation_id=args.invocation_id,
                snapshot_name=args.snapshot_name,
                keep_sandboxes=args.keep_sandboxes,
                echo_lifecycle=not args.quiet_lifecycle,
                allow_overwrite=args.overwrite,
            )
        )
    except KeyboardInterrupt as exc:
        raise SystemExit(130) from exc
    except (IntegrationUnavailableError, IntegrationContractError, ValueError) as exc:
        raise SystemExit(f"GRAVITY GAUNTLET FAILED: {exc}") from exc


if __name__ == "__main__":
    main()


__all__ = [
    "IntegrationComponents",
    "IntegrationContractError",
    "IntegrationUnavailableError",
    "LifecycleCollector",
    "LiveStateTracker",
    "build_generation_state",
    "coerce_generation_results",
    "load_integration_components",
    "run_training_demo",
    "save_generation_json",
    "save_training_state_json",
    "validate_training_rollout_contract",
    "validate_generation_seeds",
    "world_state_from_result",
]
