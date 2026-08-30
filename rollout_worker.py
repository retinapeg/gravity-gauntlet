"""Headless deterministic rollout boundary for Gravity Gauntlet.

The worker intentionally contains no physics, policy, training loop, or
Daytona SDK integration.  It calls the same ``GravityEnv`` used by the visual
demo and keeps its input/output JSON-safe so a future sandbox runner can invoke
this file directly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from itertools import repeat
from pathlib import Path
from typing import Any

from gravity_env import GravityEnv


Action = Sequence[float]
DEFAULT_MAX_STEPS = 600


def run_rollout(
    seed: int,
    actions: Iterable[Action] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    """Execute one deterministic episode using only ``GravityEnv``.

    ``actions`` is an iterable of ``[thrust_x, thrust_y]`` values.  If it is
    omitted, the ship coasts with zero thrust until the environment ends or
    ``max_steps`` is reached.  Reusing the same seed and actions reproduces the
    same rollout.
    """
    seed = int(seed)
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero")

    env = GravityEnv(seed=seed, max_steps=max_steps)
    observation, reset_info = env.reset(seed=seed)

    action_iterator = (
        iter(repeat((0.0, 0.0))) if actions is None else iter(actions)
    )
    actions_exhausted = False
    terminated = False
    truncated = False
    final_info: Mapping[str, Any] = reset_info
    transitions: list[dict[str, Any]] = []

    for _ in range(max_steps):
        try:
            requested_action = _validate_action(next(action_iterator))
        except StopIteration:
            actions_exhausted = True
            break

        next_observation, reward, terminated, truncated, info = env.step(
            requested_action
        )
        if "action" not in info:
            raise RuntimeError(
                'GravityEnv.step() info must expose its clamped "action"'
            )
        applied_action = _validate_action(info["action"])

        transitions.append(
            {
                "observation": observation,
                "requested_action": requested_action,
                "action": applied_action,
                "reward": float(reward),
                "next_observation": next_observation,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        )
        observation = next_observation
        final_info = info

        if terminated or truncated:
            break

    result = {
        "seed": seed,
        "steps": len(transitions),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(final_info.get("success", False)),
        "actions_exhausted": actions_exhausted,
        "initial_observation": (
            transitions[0]["observation"] if transitions else observation
        ),
        "final_observation": observation,
        "reset_info": reset_info,
        "info": final_info,
        "transitions": transitions,
    }
    return _json_safe(result)


def execute_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Execute a small JSON job envelope suitable for a future sandbox call.

    Expected shape::

        {"seed": 7, "max_steps": 600, "actions": [[0.0, 1.0], ...]}

    ``actions`` may be omitted to request a zero-thrust rollout.
    """
    if not isinstance(job, Mapping):
        raise ValueError("job must be a JSON object")

    return run_rollout(
        seed=job.get("seed", 0),
        actions=job.get("actions"),
        max_steps=job.get("max_steps", DEFAULT_MAX_STEPS),
    )


def _validate_action(action: Action) -> list[float]:
    if isinstance(action, (str, bytes)):
        raise ValueError("each action must contain exactly two finite numbers")
    try:
        if len(action) != 2:
            raise ValueError
        values = [float(action[0]), float(action[1])]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            "each action must contain exactly two finite numbers"
        ) from exc

    if not all(math.isfinite(value) for value in values):
        raise ValueError("each action must contain exactly two finite numbers")
    return values


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"rollout value is not JSON-safe: {type(value).__name__}")


def _load_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic GravityEnv episode without RL."
    )
    parser.add_argument(
        "--job",
        metavar="PATH",
        help='JSON job envelope path, or "-" to read it from stdin',
    )
    parser.add_argument("--seed", type=int, default=0, help="Universe seed")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of environment steps",
    )
    parser.add_argument(
        "--actions",
        metavar="PATH",
        help='JSON action-list path, or "-" to read it from stdin',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.job and args.actions:
        raise SystemExit("use either --job or --actions, not both")

    if args.job:
        result = execute_job(_load_json(args.job))
    else:
        actions = _load_json(args.actions) if args.actions else None
        result = run_rollout(args.seed, actions, args.max_steps)

    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
