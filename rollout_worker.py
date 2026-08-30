"""Headless deterministic rollout boundary for Gravity Gauntlet.

The worker intentionally contains no physics implementation, policy network
definition, training loop, or Daytona SDK integration. It calls the same
``GravityEnv`` and policy helpers used elsewhere, keeping its input/output
JSON-safe so a sandbox runner can invoke this file directly.
"""

from __future__ import annotations

import argparse
import json
import math
import operator
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gravity_env import ACTION_HOLD_STEPS, OBSERVATION_DIM, GravityEnv
from rl_policy import (
    action_index_to_vector,
    decode_policy_weights,
    make_action_generator,
    sample_action,
    sample_random_action,
)


DEFAULT_MAX_STEPS = 500


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def run_rollout(
    seed: int,
    policy_version: int = 0,
    policy_weights: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    *,
    sandbox_id: str | None = None,
) -> dict[str, Any]:
    """Run one complete categorical-policy episode using the shared physics.

    ``max_steps`` counts policy decisions. Each decision is held for several
    deterministic physics ticks inside :class:`GravityEnv`.
    """

    seed = _integer(seed, "seed", minimum=-(1 << 63))
    policy_version = _integer(policy_version, "policy_version", minimum=0)
    max_steps = _integer(max_steps, "max_steps", minimum=1)
    if policy_weights is not None and not isinstance(policy_weights, str):
        raise ValueError("policy_weights must be null or a base64 string")
    if policy_version == 0 and policy_weights is not None:
        raise ValueError("policy version 0 requires null weights")
    if policy_version > 0 and policy_weights is None:
        raise ValueError("policy version 1 or later requires encoded weights")

    model = (
        None
        if policy_weights is None
        else decode_policy_weights(policy_weights, OBSERVATION_DIM)
    )
    generator = make_action_generator(seed, policy_version)
    env = GravityEnv(
        seed=seed,
        max_steps=max_steps * ACTION_HOLD_STEPS,
    )
    observation = env.get_observation()

    observations: list[list[float]] = []
    actions: list[int] = []
    action_vectors: list[list[float]] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        observations.append(observation)
        action_index = (
            sample_random_action(generator)
            if model is None
            else sample_action(model, observation, generator=generator)
        )
        next_observation, reward, terminated, truncated, _ = env.step_discrete(
            action_index,
            hold_steps=ACTION_HOLD_STEPS,
        )
        actions.append(action_index)
        action_vectors.append(list(action_index_to_vector(action_index)))
        rewards.append(float(reward))
        observation = next_observation
        if terminated or truncated:
            break

    final_info = env.info()
    result = {
        # A real Daytona caller sets this environment variable. Local execution
        # leaves it null rather than inventing a sandbox identity.
        "sandbox_id": sandbox_id or os.environ.get("DAYTONA_SANDBOX_ID"),
        "seed": seed,
        "policy_version": policy_version,
        "reward": float(math.fsum(rewards)),
        "success": env.success,
        "termination": env.status,
        "steps": len(actions),
        "physics_steps": env.timestep,
        "universe": env.universe_dict(),
        "trajectory": env.trajectory,
        "observations": observations,
        "actions": actions,
        "action_vectors": action_vectors,
        "rewards": rewards,
        "final_observation": observation,
        "min_clearance": env.min_clearance_seen,
        "fuel_used": env.fuel_used,
        "mean_speed": final_info["mean_speed"],
        "max_speed": env.max_speed,
        "policy_mode": "seeded_random_v0" if model is None else "neural_policy",
    }
    return _json_safe(result)


def execute_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the frozen JSON envelope used by Daytona orchestration."""

    if not isinstance(job, Mapping):
        raise ValueError("job must be a JSON object")
    return run_rollout(
        seed=job.get("seed"),
        policy_version=job.get("policy_version", 0),
        policy_weights=job.get("policy_weights"),
        max_steps=job.get("max_steps", DEFAULT_MAX_STEPS),
    )


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
        description="Run one Gravity Gauntlet policy episode."
    )
    parser.add_argument(
        "--job",
        metavar="PATH",
        help='JSON job envelope path, or "-" to read it from stdin',
    )
    parser.add_argument("--seed", type=int, default=18473, help="Universe seed")
    parser.add_argument("--policy-version", type=int, default=0)
    parser.add_argument("--policy-weights", default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of held policy decisions",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.job:
            job = _load_json(args.job)
        else:
            stdin_payload = "" if sys.stdin.isatty() else sys.stdin.read()
            job = (
                json.loads(stdin_payload)
                if stdin_payload.strip()
                else {
                    "seed": args.seed,
                    "policy_version": args.policy_version,
                    "policy_weights": args.policy_weights,
                    "max_steps": args.max_steps,
                }
            )
        result = execute_job(job)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"rollout worker error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
