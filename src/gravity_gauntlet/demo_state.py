"""JSON-safe state models shared by the controller and visual demo.

The models in this module contain no physics, sandbox, or policy-training
logic.  They are deliberately small so a completed Daytona generation can be
persisted and consumed by a renderer without importing PyTorch or the Daytona
SDK.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from typing import Any, Mapping


LIFECYCLE_STATES = (
    "CREATING",
    "LIVE",
    "RUNNING",
    "SUCCESS",
    "COLLISION",
    "OUT_OF_BOUNDS",
    "TIMEOUT",
    "ERROR",
    "RESULT_COLLECTED",
)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSON output."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    """Convert simple scientific/Python values into strict JSON values.

    Daytona and policy components are expected to return JSON-safe values, but
    this boundary also accepts tuples and NumPy/PyTorch scalar-like objects.
    Non-finite numbers are rejected instead of writing non-standard JSON.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("demo state cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    raise TypeError(f"demo state value is not JSON-safe: {type(value).__name__}")


def validate_visual_trajectory(
    value: Any,
    *,
    label: str = "trajectory",
) -> list[dict[str, Any]]:
    """Return a renderer-compatible copy of one real trajectory.

    The visual loader needs at least two finite ``x``/``y`` points. Enforcing
    that same contract before a policy update prevents a generation from being
    trained and persisted only to fail when ``visual_demo.py`` reads it.
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a sequence")
    if len(value) < 2:
        raise ValueError(f"{label} must contain at least two points")

    normalized: list[dict[str, Any]] = []
    for index, point in enumerate(value):
        if not isinstance(point, Mapping):
            raise ValueError(f"{label} point {index} must be a mapping")
        missing = [field for field in ("x", "y") if field not in point]
        if missing:
            raise ValueError(
                f"{label} point {index} is missing " + ", ".join(missing)
            )
        copied = json_safe(dict(point))
        for field in ("x", "y"):
            coordinate = copied[field]
            if isinstance(coordinate, bool):
                raise ValueError(
                    f"{label} point {index} {field} must be a finite number"
                )
            try:
                number = float(coordinate)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label} point {index} {field} must be a finite number"
                ) from exc
            if not math.isfinite(number):
                raise ValueError(
                    f"{label} point {index} {field} must be a finite number"
                )
            copied[field] = number
        normalized.append(copied)
    return normalized


@dataclass(slots=True)
class WorldState:
    """The collected result and Daytona lifecycle for one seeded universe."""

    world_index: int
    seed: int
    policy_version: int
    sandbox_id: str | None = None
    status: str = "RESULT_COLLECTED"
    reward: float | None = None
    success: bool = False
    termination: str | None = None
    trajectory: list[Any] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    execution_backend: str | None = None
    min_clearance: float | None = None
    episode_length: int | None = None
    mean_speed: float | None = None
    max_speed: float | None = None
    fuel_used: float | None = None
    lifecycle: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def steps(self) -> int | None:
        """Return the worker step count under the shared JSON contract name."""

        return self.episode_length

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Keep ``episode_length`` for existing consumers while exposing the
        # literal GRAV 3 world-state contract name.
        result["steps"] = self.steps
        return json_safe(result)

    def visual_dict(self, *, generation: int | None = None) -> dict[str, Any]:
        """Return the real fields consumed by Agent 2's ghost renderer."""

        result = {
            "world_index": self.world_index,
            "seed": self.seed,
            "sandbox_id": self.sandbox_id,
            "policy_version": self.policy_version,
            "reward": self.reward,
            "success": self.success,
            "termination": self.termination,
            "trajectory": json_safe(self.trajectory),
            "actions": json_safe(self.actions),
            "execution_backend": self.execution_backend,
            "steps": self.steps,
        }
        if generation is not None:
            result["generation"] = int(generation)
        return result


@dataclass(slots=True)
class GenerationState:
    """Metrics, champion, and trajectories for one policy generation."""

    generation: int
    policy_version: int
    worlds: list[WorldState]
    average_reward: float | None
    best_reward: float | None
    worst_reward: float | None
    success_rate: float
    collision_rate: float
    average_episode_length: float | None
    best_world: int | None
    best_sandbox_id: str | None
    average_min_clearance: float | None = None
    mean_speed: float | None = None
    max_speed: float | None = None
    fuel_used: float | None = None
    next_policy_version: int | None = None
    status: str = "COMPLETE"
    timestamp: str = field(default_factory=utc_timestamp)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def world_count(self) -> int:
        return len(self.worlds)

    @property
    def policy_version_used(self) -> int:
        """Return the frozen policy version evaluated by this generation."""

        return self.policy_version

    @property
    def champion(self) -> WorldState | None:
        if self.best_world is None:
            return None
        return next(
            (world for world in self.worlds if world.world_index == self.best_world),
            None,
        )

    def champion_dict(self) -> dict[str, Any] | None:
        champion = self.champion
        if champion is None:
            return None
        return {
            "world_index": champion.world_index,
            "seed": champion.seed,
            "sandbox_id": champion.sandbox_id,
            "policy_version": champion.policy_version,
            "reward": champion.reward,
            "success": champion.success,
            "termination": champion.termination,
            "trajectory": json_safe(champion.trajectory),
            "actions": json_safe(champion.actions),
            "execution_backend": champion.execution_backend,
            "steps": champion.steps,
            "generation": self.generation,
        }

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # ``policy_version`` remains for the current visual loader; the
        # explicit alias makes the evaluated-vs-next version transition
        # unambiguous to judge-facing and external consumers.
        result["policy_version_used"] = self.policy_version_used
        result["worlds"] = [world.to_dict() for world in self.worlds]
        result["world_count"] = self.world_count
        result["champion"] = self.champion_dict()
        # Agent 2's visual loader consumes a top-level ``rollouts`` list.
        # Keep ``worlds`` as the full canonical state and expose compact views
        # of those same real trajectories—no trajectory is regenerated.
        result["rollouts"] = [
            world.visual_dict(generation=self.generation) for world in self.worlds
        ]
        return json_safe(result)


@dataclass(slots=True)
class TrainingState:
    """Compact cross-generation history for learning-progression visuals."""

    current_generation: int = -1
    current_policy_version: int = 0
    best_reward_ever: float | None = None
    total_worlds_run: int = 0
    recent_generations: list[GenerationState] = field(default_factory=list)
    history_limit: int = 12

    def __post_init__(self) -> None:
        if isinstance(self.history_limit, bool) or not isinstance(
            self.history_limit, int
        ):
            raise TypeError("history_limit must be an integer")
        if self.history_limit <= 0:
            raise ValueError("history_limit must be positive")

    def add_generation(self, generation: GenerationState) -> None:
        self.current_generation = generation.generation
        self.current_policy_version = (
            generation.next_policy_version
            if generation.next_policy_version is not None
            else generation.policy_version
        )
        self.total_worlds_run += generation.world_count
        if generation.best_reward is not None:
            if self.best_reward_ever is None:
                self.best_reward_ever = generation.best_reward
            else:
                self.best_reward_ever = max(
                    self.best_reward_ever, generation.best_reward
                )
        self.recent_generations.append(generation)
        if len(self.recent_generations) > self.history_limit:
            del self.recent_generations[: -self.history_limit]

    def ghost_history(self) -> list[dict[str, Any]]:
        """Return renderer-ready recent worlds and champions."""

        return [
            {
                "generation": generation.generation,
                "policy_version": generation.policy_version,
                "champion": generation.champion_dict(),
                "worlds": [
                    world.visual_dict(generation=generation.generation)
                    for world in generation.worlds
                ],
            }
            for generation in self.recent_generations
        ]

    def visual_rollouts(self) -> list[dict[str, Any]]:
        """Flatten recent generations for visual_demo.py --rollouts."""

        return [
            world.visual_dict(generation=generation.generation)
            for generation in self.recent_generations
            for world in generation.worlds
        ]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "current_generation": self.current_generation,
                "current_policy_version": self.current_policy_version,
                "best_reward_ever": self.best_reward_ever,
                "total_worlds_run": self.total_worlds_run,
                "recent_generations": [
                    generation.to_dict() for generation in self.recent_generations
                ],
                "ghost_history": self.ghost_history(),
                "rollouts": self.visual_rollouts(),
            }
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=True, allow_nan=False
        )


__all__ = [
    "GenerationState",
    "LIFECYCLE_STATES",
    "TrainingState",
    "WorldState",
    "json_safe",
    "validate_visual_trajectory",
    "utc_timestamp",
]
