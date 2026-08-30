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
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .demo_state import (
    GenerationState,
    TrainingState,
    WorldState,
    json_safe,
    validate_visual_trajectory,
)


DEFAULT_WORLDS = 8
DEFAULT_GENERATIONS = 1
DEFAULT_MAX_STEPS = 500
DEFAULT_BASE_SEED = 18_473
DEFAULT_INITIAL_POLICY_VERSION = 0
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


class IntegrationUnavailableError(RuntimeError):
    """Raised when one of the real integration components is unavailable."""


class IntegrationContractError(RuntimeError):
    """Raised when real component output does not satisfy the shared contract."""


@dataclass(frozen=True, slots=True)
class IntegrationComponents:
    """The one canonical trainer entry point used by the glue layer."""

    run_daytona_training: Callable[..., Any]


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

    return IntegrationComponents(run_daytona_training=run_daytona_training)


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
    if start_generation != 0 or initial_policy_version != 0:
        raise IntegrationContractError(
            "fresh controller runs must start at generation 0 / policy v0; "
            "v0 is the null-weight uniform policy and v1+ comes only from a "
            "successful trainer update"
        )
    if policy_checkpoint is not None:
        raise IntegrationContractError(
            "controller checkpoint resume is not yet part of the canonical "
            "trainer interface"
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

    _preflight_output_paths(
        runs_dir=Path(runs_dir),
        checkpoint_dir=Path(checkpoint_dir),
        generations=generations,
        allow_overwrite=bool(allow_overwrite),
    )
    training_state = TrainingState(
        current_generation=-1,
        current_policy_version=0,
    )
    lifecycle = LifecycleCollector(echo=echo_lifecycle)
    validated: dict[int, tuple[list[int], list[Mapping[str, Any]]]] = {}

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
        validated[int(policy_version)] = (seed_values, rollout_values)

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
        if next_policy_version != policy_version + 1:
            raise IntegrationContractError(
                "trainer policy version must advance exactly once per generation"
            )
        if policy_version not in validated:
            raise IntegrationContractError(
                f"trainer persisted policy v{policy_version} without validated rollouts"
            )
        seeds, validated_rollouts = validated.pop(policy_version)
        if list(rollouts) != validated_rollouts:
            raise IntegrationContractError(
                "trainer changed rollout structures between validation and persistence"
            )

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
            execution_backend="daytona",
            extra={
                "execution_backend": "daytona",
                "seed_batch": seeds,
                "trainer_checkpoint": record.get("checkpoint"),
                "training": json_safe(training_fields),
            },
        )
        incomplete_worlds = [
            world.world_index
            for world in generation_state.worlds
            if world.execution_backend != "daytona"
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
                "Daytona generation did not provide complete sandbox/lifecycle/"
                "trajectory/action proof for every world; the trained generation "
                f"was not persisted (incomplete worlds: {incomplete_worlds})"
            )

        generation_path = save_generation_json(generation_state, runs_dir)
        training_state.add_generation(generation_state)
        save_training_state_json(training_state, runs_dir)
        _print_generation_result(generation_state, generation_path)

    try:
        history = loaded.run_daytona_training(
            generations=generations,
            worlds=worlds,
            max_steps=max_steps,
            obs_dim=observation_dimension,
            base_seed=int(base_seed),
            checkpoint_dir=checkpoint_dir,
            snapshot_name=snapshot_name,
            keep_sandboxes=bool(keep_sandboxes),
            event_callback=lifecycle,
            rollout_validator=validate_before_update,
            on_generation=persist_after_update,
        )
        if inspect.isawaitable(history):
            history = await history
    except IntegrationContractError:
        raise
    except Exception as exc:
        raise IntegrationContractError(
            "canonical Daytona training failed; no local rollout fallback was "
            f"used: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(history, Sequence) or len(history) != generations:
        raise IntegrationContractError(
            "trainer did not return one completed record per requested generation"
        )
    if len(training_state.recent_generations) != generations:
        raise IntegrationContractError(
            "trainer completed without persisting every validated generation"
        )
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
    print("Encoded weights changed: yes")
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
    allow_overwrite: bool,
) -> None:
    if allow_overwrite:
        return
    targets = [
        runs_dir / f"generation_{generation:03d}.json"
        for generation in range(generations)
    ]
    targets.append(runs_dir / "training_state.json")
    targets.extend(
        checkpoint_dir / f"policy_v{version:03d}.pt"
        for version in range(generations + 1)
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
                snapshot_name=args.snapshot_name,
                keep_sandboxes=args.keep_sandboxes,
                echo_lifecycle=not args.quiet_lifecycle,
                allow_overwrite=args.overwrite,
            )
        )
    except (IntegrationUnavailableError, IntegrationContractError, ValueError) as exc:
        raise SystemExit(f"GRAVITY GAUNTLET FAILED: {exc}") from exc


if __name__ == "__main__":
    main()


__all__ = [
    "IntegrationComponents",
    "IntegrationContractError",
    "IntegrationUnavailableError",
    "LifecycleCollector",
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
