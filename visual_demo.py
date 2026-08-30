"""Colourful manual game and provenance-safe rollout replay for GravityEnv.

All simulation and collision logic remains in ``gravity_env.py``.  This file
either gathers keyboard input and calls ``GravityEnv.step()``, or renders exact
recorded samples over a freshly reconstructed matching seeded universe.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from gravity_env import ACTION_HOLD_STEPS, GravityEnv

try:
    import pygame
except ImportError:  # Allows non-visual tooling to import this module safely.
    pygame = None  # type: ignore[assignment]


WIDTH = 1200
HEIGHT = 800
FPS = 60
TRAIL_LENGTH = 650
REPLAY_POINTS_PER_SECOND = 240.0
REPLAY_END_HOLD_SECONDS = 2.8

SPACE = (3, 5, 17)
WHITE = (232, 242, 255)
MUTED = (126, 151, 187)
CYAN = (69, 225, 255)
GREEN = (85, 255, 174)
RED = (255, 70, 92)

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
    world_index: int | None = None
    termination: str | None = None
    min_clearance: float | None = None
    mean_speed: float | None = None
    max_speed: float | None = None
    provenance: str = "LOCAL REPLAY"
    daytona_verified: bool = False
    is_champion: bool = False
    champion_declared: bool = False
    source_file: str | None = None


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


def _controller_daytona_proof(container: dict[str, Any]) -> bool:
    """Recognise the real-only controller envelope without trusting an ID alone."""

    worlds = container.get("worlds")
    if container.get("status") != "COMPLETE" or not isinstance(worlds, list) or not worlds:
        return False
    if container.get("world_count") != len(worlds):
        return False
    generation_policy = container.get("policy_version")
    next_policy = container.get("next_policy_version")
    if (
        isinstance(generation_policy, bool)
        or not isinstance(generation_policy, int)
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
    }
    if not isinstance(generation_extra, dict) or not required_training_fields.issubset(
        generation_extra
    ):
        return False
    if (
        generation_extra.get("execution_backend") != "daytona"
        or not isinstance(generation_extra.get("trainer_checkpoint"), str)
        or not generation_extra["trainer_checkpoint"]
        or not isinstance(generation_extra.get("training"), dict)
        or not generation_extra["training"]
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
        if world_index is None:
            return False
        world_indices.append(world_index)
        if not isinstance(world.get("trajectory"), list) or len(world["trajectory"]) < 2:
            return False
        actions = world.get("actions")
        extra = world.get("extra")
        if (
            not isinstance(actions, list)
            or not actions
            or not isinstance(extra, dict)
            or not isinstance(extra.get("action_vectors"), list)
            or len(extra["action_vectors"]) != len(actions)
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
        states: set[str] = set()
        for event in lifecycle:
            if not isinstance(event, dict):
                return False
            state = event.get("state")
            if isinstance(state, str):
                states.add(state)
            event_id = event.get("sandbox_id")
            if event_id is not None and event_id != sandbox_id:
                return False
            event_seed = event.get("seed")
            if event_seed is not None and event_seed != seed:
                return False
        if not {"LIVE", "RUNNING", "RESULT_COLLECTED"}.issubset(states):
            return False
        expected_terminal_state = (
            "SUCCESS"
            if termination == "success"
            else "COLLISION"
            if "collision" in termination
            else termination.upper()
        )
        if expected_terminal_state not in states:
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
        champion.get("sandbox_id") == best_sandbox_id
        and champion.get("seed") == seeds[best_index]
        and champion.get("world_index") == world_indices[best_index]
        and champion.get("policy_version") == generation_policy
        and math.isclose(
            champion_reward,
            rewards[best_index],
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
        and champion.get("trajectory") == best_world.get("trajectory")
    )
    return (
        len(sandbox_ids) == len(set(sandbox_ids))
        and len(seeds) == len(set(seeds))
        and len(world_indices) == len(set(world_indices))
        and seed_batch == seeds
        and container.get("best_sandbox_id") == best_sandbox_id
        and container.get("best_world") == world_indices[best_index]
        and math.isclose(
            _finite_number(container.get("best_reward"), "best reward"),
            max(rewards),
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
        and sandbox_ids[rewards.index(max(rewards))] == best_sandbox_id
        and champion_matches
    )


def _raw_daytona_proof(container: dict[str, Any]) -> bool:
    """Validate the direct ``daytona_orchestrator --output`` envelope."""

    summary = container.get("summary")
    results = container.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list) or not results:
        return False
    if summary.get("worlds") != len(results):
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
            not isinstance(actions, list)
            or not actions
            or not isinstance(action_vectors, list)
            or len(action_vectors) != len(actions)
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
            or not isinstance(success, bool)
            or not isinstance(termination, str)
            or success != (termination == "success")
        ):
            return False
        seeds.append(seed)
        policies.add(policy)
        successes += int(success)
    best_sandbox = summary.get("best_sandbox")
    return (
        len(ids) == len(set(ids))
        and len(seeds) == len(set(seeds))
        and len(policies) == 1
        and summary.get("successful") == successes
        and best_sandbox in ids
        and math.isclose(
            _finite_number(summary.get("best_reward"), "best reward"),
            max(rewards),
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
        and ids[rewards.index(max(rewards))] == best_sandbox
    )


def _rollout_records(payload: Any) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Flatten supported artifacts while retaining their batch-level evidence."""

    if isinstance(payload, list):
        return [(record, {}) for record in payload]
    if not isinstance(payload, dict):
        raise ValueError("rollout JSON must be an object or list")

    recent = payload.get("recent_generations")
    if isinstance(recent, list) and recent:
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
        raise ValueError(f"'{next(key for key in ('worlds', 'results', 'rollouts') if key in payload)}' must be a list")

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

    context = {
        "generation": payload.get("generation"),
        "policy_version": payload.get("policy_version"),
        "next_policy_version": payload.get("next_policy_version"),
        "champion_sandbox": champion_sandbox,
        "champion_seed": champion_seed,
        "champion_world": champion_world,
        "envelope_kind": envelope_kind,
        "verified": verified,
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
        elif sandbox_id is not None:
            provenance = "UNVERIFIED RECORDED REPLAY"
        else:
            provenance = "LOCAL REPLAY"
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
                world_index=world_index,
                termination=(
                    str(rollout["termination"])
                    if rollout.get("termination") is not None
                    else None
                ),
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
            continue
        scored = [index for index in indices if attempts[index].reward is not None]
        if scored:
            champion_index = max(scored, key=lambda item: float(attempts[item].reward))
            attempts[champion_index] = replace(attempts[champion_index], is_champion=True)
    return attempts


def _xy(value: Any) -> tuple[float, float]:
    """Convert an environment position or velocity to a drawable pair."""

    return float(value[0]), float(value[1])


def _position(entity: dict[str, Any]) -> tuple[float, float]:
    return _xy(entity["position"])


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


def draw_stars(screen: Any, stars: list[tuple[int, int, int, int, float, float]], elapsed: float) -> None:
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

    points = list(trail)
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


def draw_ghost_trails(
    screen: Any,
    attempts: list[AttemptTrail],
    current_seed: int,
) -> None:
    """Render only paths recorded in this exact seeded universe."""

    assert pygame is not None
    eligible = [
        attempt
        for attempt in attempts
        if attempt.seed == current_seed and len(attempt.points) >= 2
    ]
    if not eligible:
        return

    layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for attempt in eligible:
        colour = (89, 255, 183) if attempt.is_champion else (115, 139, 185)
        alpha = 90 if attempt.is_champion else 34
        width = 2 if attempt.is_champion else 1
        pygame.draw.lines(layer, (*colour, alpha), False, attempt.points, width)

        if attempt.is_champion:
            endpoint = (round(attempt.points[-1][0]), round(attempt.points[-1][1]))
            pygame.draw.circle(layer, (*colour, 145), endpoint, 6, 2)

    screen.blit(layer, (0, 0))


def draw_champion_label(
    screen: Any,
    attempts: list[AttemptTrail],
    current_seed: int,
    label_font: Any,
) -> None:
    assert pygame is not None
    eligible = [
        attempt
        for attempt in attempts
        if attempt.seed == current_seed and attempt.is_champion and len(attempt.points) >= 2
    ]
    if not eligible:
        return
    champion = eligible[0]
    endpoint = (round(champion.points[-1][0]), round(champion.points[-1][1]))
    label = label_font.render("CURRENT CHAMPION", True, (115, 255, 195))
    label_x = max(8, min(WIDTH - label.get_width() - 8, endpoint[0] + 10))
    label_y = max(8, min(HEIGHT - label.get_height() - 8, endpoint[1] - 22))
    screen.blit(label, (label_x, label_y))


def draw_recorded_trail(screen: Any, attempt: AttemptTrail, cursor: int) -> None:
    """Reveal the exact recorded points with a long, layered glow."""

    assert pygame is not None
    end = min(len(attempt.points), max(1, cursor + 1))
    start = max(0, end - TRAIL_LENGTH)
    points = attempt.points[start:end]
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
    heading = "PARALLEL DAYTONA UNIVERSES" if active.daytona_verified else "PARALLEL RECORDED UNIVERSES"
    screen.blit(text_font.render(heading, True, WHITE), (872, 29))
    policy = f"POLICY v{active.policy_version}" if active.policy_version is not None else "POLICY N/A"
    generation = f"GEN {active.generation}" if active.generation is not None else "GEN N/A"
    trained_next = (
        f"  →  TRAINED v{active.next_policy_version}"
        if active.daytona_verified and active.next_policy_version is not None
        else ""
    )
    screen.blit(
        small_font.render(f"{generation}  •  {policy}{trained_next}", True, MUTED),
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
        pygame.draw.rect(card_layer, (*border, 220), card_layer.get_rect(), 2 if selected or attempt.is_champion else 1, border_radius=8)
        screen.blit(card_layer, card.topleft)

        if attempt.seed is not None:
            preview = _mini_world_surface(
                attempt,
                preview_envs[attempt.seed],
                active=selected,
            )
            screen.blit(preview, (card.x + 7, card.y + 8))

        world_number = attempt.world_index if attempt.world_index is not None else index + 1
        badge = "  CURRENT CHAMPION" if attempt.is_champion else "  ACTIVE" if selected else ""
        reward = f"{attempt.reward:+.2f}" if attempt.reward is not None else "N/A"
        seed_text = str(attempt.seed) if attempt.seed is not None else "N/A"
        screen.blit(small_font.render(f"WORLD {world_number:02d}{badge}", True, border), (card.x + 119, card.y + 8))
        screen.blit(small_font.render(f"SEED {seed_text}  •  RWD {reward}", True, WHITE), (card.x + 119, card.y + 28))
        screen.blit(small_font.render(_outcome_label(attempt), True, _outcome_colour(attempt)), (card.x + 119, card.y + 48))


def draw_replay_banner(
    screen: Any,
    attempt: AttemptTrail,
    fonts: tuple[Any, Any, Any],
) -> None:
    assert pygame is not None
    title_font, text_font, _ = fonts
    rect = pygame.Rect(520, 18, 320, 82)
    translucent_panel(screen, rect)
    if attempt.is_champion:
        headline = "CURRENT CHAMPION"
        colour = GREEN
    else:
        world = attempt.world_index if attempt.world_index is not None else "?"
        headline = f"PARALLEL WORLD {world}"
        colour = CYAN
    reward = f"reward {attempt.reward:+.2f}" if attempt.reward is not None else "reward N/A"
    screen.blit(title_font.render(headline, True, colour), (536, 29))
    screen.blit(text_font.render(f"{_outcome_label(attempt)}  •  {reward}", True, WHITE), (537, 65))


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
        center = (round(x), round(y))
        radius = max(3, round(float(planet["radius"])))
        fallback = PLANET_PALETTE[(seed + index * 3) % len(PLANET_PALETTE)]
        colour = tuple(planet.get("colour", planet.get("color", fallback)))
        gravity_radius = max(radius + 2, round(float(planet.get("gravity_radius", radius * 4))))
        _gravity_ring(effects, center, gravity_radius, colour)
        pygame.draw.circle(effects, (*colour, 18), center, max(radius + 18, int(radius * 1.7)))
        pygame.draw.circle(effects, (*colour, 38), center, radius + 10)

    screen.blit(effects, (0, 0))

    for index, planet in enumerate(planets):
        x, y = _position(planet)
        center = (round(x), round(y))
        radius = max(3, round(float(planet["radius"])))
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
        center = _position(asteroid)
        radius = float(asteroid["radius"])
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
    center = (round(x), round(y))
    radius = max(5, round(float(portal["radius"])))
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
    x, y = position
    radius = max(9.0, ship_radius * 1.35)
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
    assert pygame is not None
    title_font, text_font = fonts[:2]
    vx, vy = _xy(env.ship_velocity)
    speed = math.hypot(vx, vy)
    status, status_colour = status_text(env)
    completed_rewards = [
        float(attempt.reward) for attempt in attempts if attempt.reward is not None
    ]
    average_reward = (
        f"{sum(completed_rewards) / len(completed_rewards):8.2f}" if completed_rewards else "     N/A"
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
    generation_text = str(generation) if generation is not None else "N/A"
    policy_text = f"v{policy_version}" if policy_version is not None else "MANUAL"

    panel = pygame.Rect(18, 18, 445, 286)
    translucent_panel(screen, panel)
    screen.blit(title_font.render("GRAVITY GAUNTLET", True, WHITE), (34, 29))

    lines = (
        (f"SOURCE       {source_label}", MUTED),
        (f"SEED / STEP  {seed} / {env.timestep}", WHITE),
        (f"GEN / POLICY {generation_text} / {policy_text}", WHITE),
        (f"CURRENT RWD  {env.episode_reward:8.2f}", WHITE),
        (f"ALL AVG/BEST {average_reward} / {best_reward}", WHITE),
        (f"ALL SUCCESS  {success_rate}  ({len(known_outcomes)} outcomes)", WHITE),
        (f"VELOCITY     ({vx:6.1f}, {vy:6.1f})", WHITE),
        (f"SPEED        {speed:8.2f}", WHITE),
        (f"TARGET       {distance_to_portal(env):8.1f} px", WHITE),
        (f"MIN CLEAR    {clearance_seen:8.2f} px", WHITE if clearance_seen >= 58 else RED),
        (f"STATUS       {status}", status_colour),
    )
    for index, (text, colour) in enumerate(lines):
        screen.blit(text_font.render(text, True, colour), (35, 61 + index * 20))

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
    title_font, text_font, small_font = fonts
    group = _attempt_group(attempts, attempt)
    scored = [float(item.reward) for item in group if item.reward is not None]
    outcomes = [item.success for item in group if item.success is not None]
    velocity = _recorded_velocity(attempt, cursor)
    speed = math.hypot(*velocity) if velocity is not None else None
    clearance = _recorded_clearance(attempt, cursor)
    reward_so_far = _recorded_reward_so_far(attempt, cursor)
    finished = cursor >= len(attempt.points) - 1
    status = _outcome_label(attempt) if finished else "RECORDED FLIGHT IN PROGRESS"
    status_colour = _outcome_colour(attempt) if finished else CYAN
    source_colour = GREEN if attempt.daytona_verified else (255, 190, 74)
    artifact = Path(attempt.source_file).name if attempt.source_file else "N/A"
    sandbox_id = attempt.sandbox_id or "N/A"
    if len(sandbox_id) > 31:
        sandbox_id = f"{sandbox_id[:14]}…{sandbox_id[-14:]}"
    reward_text = f"{attempt.reward:+.2f}" if attempt.reward is not None else "N/A"
    current_reward_text = f"{reward_so_far:+.2f}" if reward_so_far is not None else "N/A"
    average_reward = f"{sum(scored) / len(scored):+.2f}" if scored else "N/A"
    best_reward = f"{max(scored):+.2f}" if scored else "N/A"
    success_rate = (
        f"{100.0 * sum(bool(value) for value in outcomes) / len(outcomes):.1f}%"
        if outcomes
        else "N/A"
    )
    speed_text = f"{speed:.2f}" if speed is not None else "N/A"
    mean_speed_text = f"{attempt.mean_speed:.2f}" if attempt.mean_speed is not None else "N/A"
    clearance_text = f"{clearance:.2f}" if clearance is not None else "N/A"
    min_clearance_text = f"{attempt.min_clearance:.2f}" if attempt.min_clearance is not None else "N/A"
    world_number = attempt.world_index if attempt.world_index is not None else group.index(attempt) + 1
    generation_text = str(attempt.generation) if attempt.generation is not None else "N/A"
    policy_text = f"v{attempt.policy_version}" if attempt.policy_version is not None else "N/A"
    sample_step = cursor
    if attempt.trajectory and attempt.trajectory[min(cursor, len(attempt.trajectory) - 1)].get("step") is not None:
        sample_step = int(attempt.trajectory[min(cursor, len(attempt.trajectory) - 1)]["step"])

    panel = pygame.Rect(18, 18, 482, 347)
    translucent_panel(screen, panel)
    screen.blit(title_font.render("GRAVITY GAUNTLET // REPLAY", True, WHITE), (34, 29))
    lines = (
        (f"SOURCE       {attempt.provenance}", source_colour),
        (f"ARTIFACT     {artifact}", MUTED),
        (f"SANDBOX      {sandbox_id}", WHITE if attempt.sandbox_id else MUTED),
        (f"WORLD / SEED {world_number}/{len(group)} / {attempt.seed if attempt.seed is not None else 'N/A'}", WHITE),
        (f"GEN / POLICY {generation_text} / {policy_text}", WHITE),
        (f"TOTAL REWARD {reward_text}   LIVE {current_reward_text}", WHITE),
        (f"GEN AVG/BEST {average_reward} / {best_reward}", WHITE),
        (f"GEN SUCCESS  {success_rate}  ({len(outcomes)} outcomes)", WHITE),
        (f"SPEED        {speed_text}   MEAN {mean_speed_text}", WHITE),
        (f"CLEARANCE    {clearance_text}   MIN {min_clearance_text}", WHITE),
        (
            f"NEXT / THRUST  {f'v{attempt.next_policy_version}' if attempt.next_policy_version is not None else 'N/A'} / "
            f"{'RECORDED' if attempt.action_vectors else 'NOT PRESENT'}",
            WHITE if attempt.next_policy_version is not None or attempt.action_vectors else MUTED,
        ),
        (f"PLAYBACK     STEP {sample_step}  POINT {cursor + 1}/{len(attempt.points)}", WHITE),
        (f"STATUS       {status}", status_colour),
    )
    for index, (text, colour) in enumerate(lines):
        screen.blit(text_font.render(text, True, colour), (35, 61 + index * 21))

    controls = pygame.Rect(18, HEIGHT - 61, 610, 43)
    translucent_panel(screen, controls)
    hint = "R  REPLAY THIS RUN     N  NEXT RECORDED UNIVERSE     ESC  EXIT"
    screen.blit(small_font.render(hint, True, MUTED), (34, HEIGHT - 47))


def draw_end_overlay(screen: Any, env: GravityEnv, fonts: tuple[Any, ...], elapsed: float) -> None:
    assert pygame is not None
    if not env.done:
        return

    title_font, text_font = fonts[:2]
    center = tuple(round(value) for value in env.ship_position)
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
            inner = (center[0] + round(math.cos(angle) * 18), center[1] + round(math.sin(angle) * 18))
            outer = (center[0] + round(math.cos(angle) * 52), center[1] + round(math.sin(angle) * 52))
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

    screen.blit(effects, (0, 0))
    banner = pygame.Rect(WIDTH // 2 - 285, HEIGHT // 2 - 62, 570, 124)
    translucent_panel(screen, banner)
    title = title_font.render(headline, True, colour)
    subtitle = text_font.render(detail, True, WHITE)
    screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 19)))
    screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 24)))


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
    impact = (
        max(10, min(WIDTH - 10, round(position[0]))),
        max(10, min(HEIGHT - 10, round(position[1]))),
    )
    portal = (round(portal_position[0]), round(portal_position[1]))
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

    screen.blit(effects, (0, 0))
    banner = pygame.Rect(WIDTH // 2 - 315, HEIGHT // 2 - 65, 630, 130)
    translucent_panel(screen, banner)
    title = title_font.render(headline, True, colour)
    subtitle = text_font.render(detail, True, WHITE)
    screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20)))
    screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 26)))


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
        provenance="LOCAL MANUAL",
    )


def _source_label(attempts: list[AttemptTrail], current_seed: int) -> str:
    del attempts, current_seed
    return "LOCAL MANUAL"


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
        if attempt.daytona_verified:
            raise ValueError(
                f"verified Daytona replay for seed {attempt.seed} has no recorded universe"
            )
        return env
    if env.universe_dict() != attempt.universe:
        raise ValueError(
            f"recorded universe for seed {attempt.seed} does not match the current "
            "GravityEnv; rebuild and verify the Daytona snapshot before replay"
        )
    return env


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
        raise SystemExit("Pygame is required. Install the project dependencies, then run python visual_demo.py")

    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        attempts = list(rollout_trails or [])
        replay_mode = bool(attempts)
        pygame.display.set_caption(
            "Gravity Gauntlet — Recorded Daytona Experience"
            if replay_mode and all(item.daytona_verified for item in attempts)
            else "Gravity Gauntlet — Recorded Replay"
            if replay_mode
            else "Gravity Gauntlet — 2D Space MVP"
        )
        clock = pygame.time.Clock()
        fonts = (
            pygame.font.SysFont("menlo", 24, bold=True),
            pygame.font.SysFont("menlo", 15),
            pygame.font.SysFont("menlo", 12, bold=True),
        )

        replay_index = _initial_replay_index(attempts, seed) if replay_mode else 0
        active_attempt = attempts[replay_index] if replay_mode else None
        current_seed = int(active_attempt.seed) if active_attempt is not None else int(seed or 7)
        preview_envs = {
            int(attempt.seed): _replay_environment(attempt)
            for attempt in attempts
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
                            replay_index = (replay_index + 1) % len(attempts)
                            active_attempt = attempts[replay_index]
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
            draw_ghost_trails(screen, attempts, current_seed)
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
            draw_champion_label(screen, attempts, current_seed, fonts[1])
            if replay_mode:
                assert active_attempt is not None
                finished = cursor >= len(active_attempt.points) - 1
                draw_replay_hud(screen, active_attempt, attempts, cursor, fonts)
                draw_parallel_universes(
                    screen,
                    attempts,
                    active_attempt,
                    preview_envs,
                    fonts,
                )
                draw_replay_banner(screen, active_attempt, fonts)
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
                        replay_cursor + REPLAY_POINTS_PER_SECOND / FPS,
                    )
                    replay_end_frames = 0
                else:
                    replay_end_frames += 1
                    if (
                        len(attempts) > 1
                        and replay_end_frames >= round(REPLAY_END_HOLD_SECONDS * FPS)
                    ):
                        replay_index = (replay_index + 1) % len(attempts)
                        active_attempt = attempts[replay_index]
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
    parser = argparse.ArgumentParser(description="Play the deterministic Gravity Gauntlet visual MVP.")
    parser.add_argument(
        "--seed",
        type=int,
        help="universe seed (default: loaded champion's seed, otherwise 7)",
    )
    parser.add_argument(
        "--rollouts",
        type=Path,
        help="optional worker/controller JSON to replay over seed-matched universes",
    )
    parser.add_argument("--generation", type=int, help="real generation number to display")
    parser.add_argument("--policy-version", type=int, help="real policy version to display")
    args = parser.parse_args()
    try:
        rollout_trails = load_rollout_trails(args.rollouts) if args.rollouts else []
        if rollout_trails:
            rollout_trails = _select_replay_group(
                rollout_trails,
                generation=args.generation,
                policy_version=args.policy_version,
            )
            replay_index = _initial_replay_index(rollout_trails, args.seed)
            selected = rollout_trails[replay_index]
            seed = int(selected.seed)
            generation = selected.generation
            policy_version = selected.policy_version
        else:
            seed = args.seed if args.seed is not None else 7
            generation = args.generation
            policy_version = args.policy_version
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"cannot load rollout replay: {exc}")
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


if __name__ == "__main__":
    main()
