"""Daytona-only JSON contract bridge around the canonical rollout worker.

This module contains no physics, policy definition, or training code. It runs
``rollout_worker.execute_job`` inside a Daytona sandbox, preserves the worker's
JSON-safe result (including categorical actions and rich trajectory points),
and adds the real sandbox identity supplied by the controller.
"""

from __future__ import annotations

import argparse
import json
import math
import operator
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rollout_worker import execute_job


TERMINATION_NAMES = {
    "portal": "success",
    "success": "success",
    "collision_planet": "planet_collision",
    "collision_asteroid": "asteroid_collision",
    "planet_collision": "planet_collision",
    "asteroid_collision": "asteroid_collision",
    "out_of_bounds": "out_of_bounds",
    "timeout": "timeout",
}


class DaytonaWorkerContractError(RuntimeError):
    """Raised when a job or worker result violates the sandbox contract."""


def execute_daytona_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one canonical rollout and return the frozen Daytona envelope."""

    normalized_job = _validate_job(job)
    raw_result = execute_job(normalized_job)
    if not isinstance(raw_result, Mapping):
        raise DaytonaWorkerContractError("rollout worker must return a JSON object")

    _matching_integer(raw_result, "seed", normalized_job["seed"])
    _matching_integer(
        raw_result,
        "policy_version",
        normalized_job["policy_version"],
    )
    worker_sandbox_id = raw_result.get("sandbox_id")
    if worker_sandbox_id is not None and (
        worker_sandbox_id != normalized_job["sandbox_id"]
    ):
        raise DaytonaWorkerContractError(
            "worker sandbox_id conflicts with the real Daytona sandbox"
        )

    result = dict(raw_result)
    worker_termination = result.get("termination")
    if worker_termination not in TERMINATION_NAMES:
        raise DaytonaWorkerContractError(
            f"unknown worker termination status: {worker_termination!r}"
        )
    normalized_termination = TERMINATION_NAMES[str(worker_termination)]
    if normalized_termination != worker_termination:
        result["worker_termination"] = worker_termination

    # execute_job intentionally knows nothing about Daytona. The controller
    # writes the real sandbox.id into the job, and this bridge is the only
    # layer allowed to inject that identity into the worker result.
    result.update(
        {
            "sandbox_id": normalized_job["sandbox_id"],
            "seed": normalized_job["seed"],
            "policy_version": normalized_job["policy_version"],
            "termination": normalized_termination,
        }
    )
    _validate_result_envelope(result)
    return result


def _validate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(job, Mapping):
        raise DaytonaWorkerContractError("job must be a JSON object")

    seed = _integer_field(job, "seed")
    policy_version = _integer_field(job, "policy_version", minimum=0)
    max_steps = _integer_field(job, "max_steps", minimum=1)
    sandbox_id = job.get("sandbox_id")
    if not isinstance(sandbox_id, str) or not sandbox_id.strip():
        raise DaytonaWorkerContractError("job requires a real sandbox_id")

    policy_weights = job.get("policy_weights")
    if policy_version == 0 and policy_weights is not None:
        raise DaytonaWorkerContractError("policy version 0 requires null weights")
    if policy_version > 0 and (
        not isinstance(policy_weights, str) or not policy_weights.strip()
    ):
        raise DaytonaWorkerContractError(
            "trained policies require JSON-safe encoded string weights"
        )

    return {
        **dict(job),
        "sandbox_id": sandbox_id,
        "seed": seed,
        "policy_version": policy_version,
        "policy_weights": policy_weights,
        "max_steps": max_steps,
    }


def _validate_result_envelope(result: Mapping[str, Any]) -> None:
    required = (
        "sandbox_id",
        "seed",
        "policy_version",
        "reward",
        "success",
        "termination",
        "steps",
        "physics_steps",
        "trajectory",
        "observations",
        "actions",
        "rewards",
    )
    missing = [field for field in required if field not in result]
    if missing:
        raise DaytonaWorkerContractError(
            "worker result missing fields: " + ", ".join(missing)
        )
    if not isinstance(result["sandbox_id"], str) or not result["sandbox_id"]:
        raise DaytonaWorkerContractError("sandbox_id must be a non-empty string")
    _finite_number(result["reward"], "reward")
    if not isinstance(result["success"], bool):
        raise DaytonaWorkerContractError("success must be boolean")
    if not isinstance(result["termination"], str) or not result["termination"]:
        raise DaytonaWorkerContractError("termination must be a non-empty string")
    if result["success"] != (result["termination"] == "success"):
        raise DaytonaWorkerContractError(
            "success must agree with the normalized termination"
        )

    steps = _integer_value(result["steps"], "steps", minimum=1)
    physics_steps = _integer_value(
        result["physics_steps"], "physics_steps", minimum=1
    )
    for field in ("trajectory", "observations", "actions", "rewards"):
        if not isinstance(result[field], list):
            raise DaytonaWorkerContractError(f"{field} must be a list")
    if not result["trajectory"]:
        raise DaytonaWorkerContractError("trajectory must not be empty")
    if len(result["trajectory"]) != physics_steps + 1:
        raise DaytonaWorkerContractError(
            "trajectory must contain the initial point plus one point per "
            "physics step"
        )
    if len(result["actions"]) != len(result["rewards"]):
        raise DaytonaWorkerContractError("actions and rewards must have equal length")
    if steps != len(result["actions"]):
        raise DaytonaWorkerContractError("steps must match the action count")
    for action in result["actions"]:
        action_index = _integer_value(action, "action")
        if not 0 <= action_index <= 8:
            raise DaytonaWorkerContractError("categorical actions must be in [0, 8]")
    reward_values = [
        _finite_number(reward, "reward") for reward in result["rewards"]
    ]
    if not math.isclose(
        _finite_number(result["reward"], "reward"),
        math.fsum(reward_values),
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ):
        raise DaytonaWorkerContractError(
            "top-level reward must equal the sum of per-step rewards"
        )

    _validate_observations(result["observations"], steps)
    if "action_vectors" in result:
        action_vectors = _step_list(
            result["action_vectors"],
            "action_vectors",
            steps,
        )
        for vector in action_vectors:
            _numeric_pair(vector, "action vector")
    for point in result["trajectory"]:
        if not isinstance(point, Mapping):
            raise DaytonaWorkerContractError("trajectory points must be JSON objects")
        for field in ("x", "y", "vx", "vy"):
            if field not in point:
                raise DaytonaWorkerContractError(
                    f"trajectory point is missing {field}"
                )
            _finite_number(point[field], f"trajectory {field}")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DaytonaWorkerContractError("worker result must be JSON-safe") from exc


def _matching_integer(
    result: Mapping[str, Any],
    field: str,
    expected: int,
) -> None:
    if field not in result:
        raise DaytonaWorkerContractError(f"worker result is missing {field}")
    actual = _integer_value(result[field], field)
    if actual != expected:
        raise DaytonaWorkerContractError(
            f"worker {field} {actual} does not match requested {expected}"
        )


def _step_list(value: Any, name: str, steps: int) -> list[Any]:
    if not isinstance(value, list):
        raise DaytonaWorkerContractError(f"{name} must be a list")
    if len(value) != steps:
        raise DaytonaWorkerContractError(f"{name} must contain one entry per step")
    return value


def _numeric_pair(value: Any, name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise DaytonaWorkerContractError(f"{name} must contain two numbers")
    _finite_number(value[0], name)
    _finite_number(value[1], name)


def _validate_observations(value: Any, steps: int) -> None:
    observations = _step_list(value, "observations", steps)
    observation_dimension: int | None = None
    for observation in observations:
        if not isinstance(observation, list) or not observation:
            raise DaytonaWorkerContractError(
                "observations must be non-empty numeric lists"
            )
        if observation_dimension is None:
            observation_dimension = len(observation)
        elif len(observation) != observation_dimension:
            raise DaytonaWorkerContractError(
                "observations must share one fixed dimension"
            )
        for item in observation:
            _finite_number(item, "observation")


def _integer_field(
    value: Mapping[str, Any],
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if name not in value:
        raise DaytonaWorkerContractError(f"job requires {name}")
    return _integer_value(value[name], name, minimum=minimum)


def _integer_value(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise DaytonaWorkerContractError(f"{name} must be an integer")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise DaytonaWorkerContractError(f"{name} must be an integer") from exc
    if minimum is not None and integer < minimum:
        raise DaytonaWorkerContractError(f"{name} must be at least {minimum}")
    return integer


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DaytonaWorkerContractError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DaytonaWorkerContractError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise DaytonaWorkerContractError(f"{name} must be finite")
    return number


def _load_job(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Gravity Gauntlet Daytona job")
    parser.add_argument("--job", default="-", help='job JSON path, or "-" for stdin')
    parser.add_argument(
        "--output",
        help="optional result JSON path; stdout is used when omitted",
    )
    args = parser.parse_args()
    result = execute_daytona_job(_load_job(args.job))
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
