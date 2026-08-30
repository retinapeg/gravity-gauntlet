"""Run Gravity Gauntlet physics concurrently inside real Daytona sandboxes.

There is deliberately no local rollout fallback in this module.  Unit tests
may replace the Daytona SDK objects with fakes, but every production call to
``run_generation`` requires Daytona credentials and creates Daytona sandboxes.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import operator
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams
except ImportError:  # A clear runtime error is raised by _require_daytona().
    AsyncDaytona = None  # type: ignore[assignment]
    CreateSandboxFromSnapshotParams = None  # type: ignore[assignment]


DEFAULT_SNAPSHOT_NAME = "gravity-gauntlet-worker-v2"
DEFAULT_BASE_SEED = 18_473
DEFAULT_WORLDS = 8
DEFAULT_MAX_STEPS = 500
REMOTE_WORK_DIR = "gravity-gauntlet-worker"
REMOTE_JOB_PATH = f"{REMOTE_WORK_DIR}/daytona_job.json"
REMOTE_RESULT_PATH = f"{REMOTE_WORK_DIR}/daytona_result.json"
WORKER_COMMAND = (
    "python3 daytona_worker_entry.py --job daytona_job.json "
    "--output daytona_result.json 2>&1"
)
SANDBOX_CREATE_TIMEOUT = 120
WORKER_TIMEOUT = 300
SANDBOX_DELETE_TIMEOUT = 120
ROLLOUT_TTL_MINUTES = 10

EventCallback = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class DaytonaConfigurationError(RuntimeError):
    """Raised when the SDK or environment credentials are unavailable."""


class DaytonaExecutionError(RuntimeError):
    """Raised when a real sandbox cannot complete a valid rollout."""


class DaytonaGenerationError(DaytonaExecutionError):
    """Aggregate failure raised after every concurrent world has settled."""

    def __init__(
        self,
        failures: list[dict[str, Any]],
        partial_results: list[dict[str, Any]],
    ) -> None:
        self.failures = failures
        self.partial_results = partial_results
        details = "; ".join(
            f"world {item['world']} seed {item['seed']}: {item['error']}"
            for item in failures
        )
        super().__init__(f"{len(failures)} Daytona world(s) failed: {details}")


async def run_world(
    daytona: Any,
    *,
    snapshot_name: str,
    world_index: int,
    seed: int,
    policy_version: int,
    policy_weights: str | None,
    max_steps: int,
    event_callback: EventCallback | None = None,
    keep_sandbox: bool = False,
) -> dict[str, Any]:
    """Create one real sandbox, execute its worker, collect, and clean up."""

    _require_sdk_types()
    world_index = int(world_index)
    seed = int(seed)
    sandbox: Any | None = None
    sandbox_id: str | None = None
    result: dict[str, Any] | None = None
    failure: Exception | None = None
    cleanup_failure: Exception | None = None

    await _emit_event(
        event_callback,
        world=world_index,
        seed=seed,
        state="CREATING",
        sandbox_id=None,
    )

    try:
        params = CreateSandboxFromSnapshotParams(
            snapshot=snapshot_name,
            language="python",
            labels={
                "application": "gravity-gauntlet",
                "purpose": "parallel-rollout",
                "world": str(world_index),
                "policy-version": str(policy_version),
            },
            auto_delete_interval=60,
            ttl_minutes=ROLLOUT_TTL_MINUTES,
        )
        sandbox = await daytona.create(params, timeout=SANDBOX_CREATE_TIMEOUT)
        sandbox_id = _real_sandbox_id(sandbox)
        await _emit_event(
            event_callback,
            world=world_index,
            seed=seed,
            state="LIVE",
            sandbox_id=sandbox_id,
        )

        job = {
            "sandbox_id": sandbox_id,
            "seed": seed,
            "policy_version": int(policy_version),
            "policy_weights": policy_weights,
            "max_steps": int(max_steps),
        }
        await _upload_json(sandbox, REMOTE_JOB_PATH, job)
        await _emit_event(
            event_callback,
            world=world_index,
            seed=seed,
            state="RUNNING",
            sandbox_id=sandbox_id,
        )

        result = await _execute_worker_job(sandbox)
        _validate_worker_result(
            result,
            sandbox_id=sandbox_id,
            seed=seed,
            policy_version=policy_version,
        )
        terminal_state = _lifecycle_terminal_state(result)
        await _emit_event(
            event_callback,
            world=world_index,
            seed=seed,
            state=terminal_state,
            sandbox_id=sandbox_id,
            reward=float(result["reward"]),
            termination=result["termination"],
        )
        await _emit_event(
            event_callback,
            world=world_index,
            seed=seed,
            state="RESULT_COLLECTED",
            sandbox_id=sandbox_id,
            reward=float(result["reward"]),
        )
    except Exception as exc:
        failure = exc
        await _emit_error_safely(
            event_callback,
            world=world_index,
            seed=seed,
            sandbox_id=sandbox_id,
            error=exc,
        )
    finally:
        if sandbox is not None and not keep_sandbox:
            try:
                await sandbox.delete(timeout=SANDBOX_DELETE_TIMEOUT, wait=True)
            except Exception as exc:
                cleanup_failure = exc
                await _emit_error_safely(
                    event_callback,
                    world=world_index,
                    seed=seed,
                    sandbox_id=sandbox_id,
                    error=DaytonaExecutionError(
                        f"sandbox cleanup failed: {type(exc).__name__}: {exc}"
                    ),
                )

    if failure is not None and cleanup_failure is not None:
        raise DaytonaExecutionError(
            f"world {world_index} failed in Daytona: "
            f"{type(failure).__name__}: {failure}; sandbox {sandbox_id} "
            f"cleanup also failed: {type(cleanup_failure).__name__}: "
            f"{cleanup_failure}"
        ) from failure
    if failure is not None:
        if isinstance(failure, DaytonaExecutionError):
            raise failure
        raise DaytonaExecutionError(
            f"world {world_index} failed in Daytona: "
            f"{type(failure).__name__}: {failure}"
        ) from failure
    if cleanup_failure is not None:
        raise DaytonaExecutionError(
            f"world {world_index} result was collected, but sandbox "
            f"{sandbox_id} could not be deleted: {cleanup_failure}"
        ) from cleanup_failure
    if result is None:
        raise DaytonaExecutionError(f"world {world_index} returned no result")
    return result


async def run_generation(
    *,
    policy_version: int,
    policy_weights: str | None,
    seeds: list[int],
    max_steps: int = DEFAULT_MAX_STEPS,
    snapshot_name: str = DEFAULT_SNAPSHOT_NAME,
    event_callback: EventCallback | None = None,
    keep_sandboxes: bool = False,
) -> list[dict[str, Any]]:
    """Fan one frozen policy out to all requested Daytona worlds concurrently."""

    _require_daytona()
    policy_version, policy_weights, seed_values, max_steps = _validate_generation(
        policy_version, policy_weights, seeds, max_steps
    )
    if not isinstance(snapshot_name, str) or not snapshot_name.strip():
        raise ValueError("snapshot_name cannot be empty")

    try:
        async with AsyncDaytona() as daytona:  # type: ignore[misc,operator]
            outcomes = await asyncio.gather(
                *(
                    run_world(
                        daytona,
                        snapshot_name=snapshot_name,
                        world_index=index,
                        seed=seed,
                        policy_version=policy_version,
                        policy_weights=policy_weights,
                        max_steps=max_steps,
                        event_callback=event_callback,
                        keep_sandbox=keep_sandboxes,
                    )
                    for index, seed in enumerate(seed_values, start=1)
                ),
                return_exceptions=True,
            )
    except DaytonaExecutionError:
        raise
    except Exception as exc:
        raise DaytonaExecutionError(
            f"Daytona client failed before the generation settled: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (seed, outcome) in enumerate(zip(seed_values, outcomes), start=1):
        if isinstance(outcome, BaseException):
            failures.append(
                {
                    "world": index,
                    "seed": seed,
                    "error": f"{type(outcome).__name__}: {outcome}",
                }
            )
        else:
            results.append(outcome)

    if failures:
        raise DaytonaGenerationError(failures, results)
    _validate_unique_sandbox_ids(results)
    return results


async def _upload_json(sandbox: Any, remote_path: str, payload: Mapping[str, Any]) -> None:
    try:
        content = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DaytonaExecutionError("sandbox job is not JSON-safe") from exc
    await sandbox.fs.upload_file(content, remote_path)


async def _execute_worker_job(sandbox: Any) -> dict[str, Any]:
    execution = await sandbox.process.exec(
        WORKER_COMMAND,
        cwd=REMOTE_WORK_DIR,
        timeout=WORKER_TIMEOUT,
    )
    exit_code = getattr(execution, "exit_code", None)
    if exit_code != 0:
        tail = str(getattr(execution, "result", "") or "").strip()[-1200:]
        raise DaytonaExecutionError(
            f"sandbox worker exited with code {exit_code}: {tail}"
        )

    payload = await sandbox.fs.download_file(REMOTE_RESULT_PATH)
    if not isinstance(payload, (bytes, bytearray)):
        raise DaytonaExecutionError(
            "sandbox result download did not return bytes"
        )
    try:
        parsed = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DaytonaExecutionError(
            "sandbox result file is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise DaytonaExecutionError("sandbox worker result must be a JSON object")
    return parsed


def _validate_worker_result(
    result: Mapping[str, Any],
    *,
    sandbox_id: str,
    seed: int,
    policy_version: int,
) -> None:
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
        raise DaytonaExecutionError(
            "worker result missing fields: " + ", ".join(missing)
        )
    if result["sandbox_id"] != sandbox_id:
        raise DaytonaExecutionError(
            "worker sandbox_id does not match the real Daytona sandbox"
        )
    if _integer_value(result["seed"], "seed") != seed:
        raise DaytonaExecutionError("worker seed does not match the requested seed")
    if _integer_value(result["policy_version"], "policy_version") != policy_version:
        raise DaytonaExecutionError("worker policy version does not match the job")
    _finite_number(result["reward"], "reward")
    if not isinstance(result["success"], bool):
        raise DaytonaExecutionError("worker success must be boolean")
    if not isinstance(result["termination"], str) or not result["termination"]:
        raise DaytonaExecutionError("worker termination must be a non-empty string")
    if result["success"] != (result["termination"] == "success"):
        raise DaytonaExecutionError(
            "worker success does not agree with termination"
        )
    steps = _integer_value(result["steps"], "steps", minimum=1)
    physics_steps = _integer_value(
        result["physics_steps"], "physics_steps", minimum=1
    )
    for field in ("trajectory", "observations", "actions", "rewards"):
        if not isinstance(result[field], list):
            raise DaytonaExecutionError(f"worker {field} must be a list")
    if not result["trajectory"]:
        raise DaytonaExecutionError("worker trajectory must not be empty")
    if len(result["trajectory"]) != physics_steps + 1:
        raise DaytonaExecutionError(
            "worker trajectory must contain the initial point plus one point "
            "per physics step"
        )
    if len(result["actions"]) != len(result["rewards"]):
        raise DaytonaExecutionError("worker actions and rewards lengths differ")
    if steps != len(result["actions"]):
        raise DaytonaExecutionError("worker steps does not match action count")
    for action in result["actions"]:
        action_index = _integer_value(action, "action")
        if not 0 <= action_index <= 8:
            raise DaytonaExecutionError("worker action must be in [0, 8]")
    reward_values = [
        _finite_number(reward, "reward") for reward in result["rewards"]
    ]
    if not math.isclose(
        _finite_number(result["reward"], "reward"),
        math.fsum(reward_values),
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ):
        raise DaytonaExecutionError(
            "worker reward does not equal the per-step reward sum"
        )
    _validate_observations(result["observations"], steps)
    if "action_vectors" in result:
        if not isinstance(result["action_vectors"], list) or len(
            result["action_vectors"]
        ) != steps:
            raise DaytonaExecutionError(
                "worker action_vectors must contain one entry per step"
            )
    for point in result["trajectory"]:
        if not isinstance(point, Mapping):
            raise DaytonaExecutionError(
                "worker trajectory points must be JSON objects"
            )
        for field in ("x", "y", "vx", "vy"):
            if field not in point:
                raise DaytonaExecutionError(
                    f"worker trajectory point is missing {field}"
                )
            _finite_number(point[field], f"trajectory {field}")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DaytonaExecutionError("worker result is not JSON-safe") from exc


def _lifecycle_terminal_state(result: Mapping[str, Any]) -> str:
    termination = str(result["termination"])
    if bool(result["success"]) or termination == "success":
        return "SUCCESS"
    if termination in {
        "collision",
        "planet_collision",
        "asteroid_collision",
        "collision_planet",
        "collision_asteroid",
    }:
        return "COLLISION"
    if termination == "out_of_bounds":
        return "OUT_OF_BOUNDS"
    if termination == "timeout":
        return "TIMEOUT"
    raise DaytonaExecutionError(f"unknown termination state: {termination!r}")


def _validate_unique_sandbox_ids(results: Sequence[Mapping[str, Any]]) -> None:
    sandbox_ids = [result.get("sandbox_id") for result in results]
    if len(sandbox_ids) != len(set(sandbox_ids)):
        raise DaytonaExecutionError(
            "Daytona returned duplicate sandbox IDs for different worlds"
        )


async def _emit_event(
    callback: EventCallback | None,
    *,
    world: int,
    seed: int,
    state: str,
    sandbox_id: str | None,
    **extra: Any,
) -> None:
    if callback is None:
        return
    event = {
        "world": int(world),
        "seed": int(seed),
        "state": state,
        "sandbox_id": sandbox_id,
        **extra,
    }
    callback_result = callback(event)
    if inspect.isawaitable(callback_result):
        await callback_result


async def _emit_error_safely(
    callback: EventCallback | None,
    *,
    world: int,
    seed: int,
    sandbox_id: str | None,
    error: Exception,
) -> None:
    try:
        await _emit_event(
            callback,
            world=world,
            seed=seed,
            state="ERROR",
            sandbox_id=sandbox_id,
            error=f"{type(error).__name__}: {error}",
        )
    except Exception:
        # A broken UI callback must not prevent sandbox cleanup.
        pass


def _validate_generation(
    policy_version: int,
    policy_weights: str | None,
    seeds: Sequence[int],
    max_steps: int,
) -> tuple[int, str | None, list[int], int]:
    policy_version = _integer_value(
        policy_version,
        "policy_version",
        minimum=0,
        error_type=ValueError,
    )
    max_steps = _integer_value(
        max_steps,
        "max_steps",
        minimum=1,
        error_type=ValueError,
    )
    if policy_version == 0 and policy_weights is not None:
        raise ValueError("policy version 0 requires null weights")
    if policy_version > 0 and (
        not isinstance(policy_weights, str) or not policy_weights.strip()
    ):
        raise ValueError("trained policies require encoded string weights")
    seed_values = [
        _integer_value(seed, "seed", error_type=ValueError) for seed in seeds
    ]
    if not seed_values:
        raise ValueError("at least one seed is required")
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("each Daytona world requires a different seed")
    return policy_version, policy_weights, seed_values, max_steps


def _require_daytona() -> None:
    _require_sdk_types()
    try:
        api_key = os.environ["DAYTONA_API_KEY"]
    except KeyError as exc:
        raise DaytonaConfigurationError(
            "DAYTONA_API_KEY is required; local rollout fallback is disabled"
        ) from exc
    if not api_key:
        raise DaytonaConfigurationError(
            "DAYTONA_API_KEY is required; local rollout fallback is disabled"
        )


def _require_sdk_types() -> None:
    if AsyncDaytona is None or CreateSandboxFromSnapshotParams is None:
        raise DaytonaConfigurationError(
            "Daytona SDK is unavailable on the host; install daytona==0.207.0"
        )


def _real_sandbox_id(sandbox: Any) -> str:
    sandbox_id = getattr(sandbox, "id", None)
    if not isinstance(sandbox_id, str) or not sandbox_id.strip():
        raise DaytonaExecutionError("Daytona returned a sandbox without a real id")
    return sandbox_id


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DaytonaExecutionError(f"worker {name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DaytonaExecutionError(f"worker {name} must be numeric") from exc
    if not math.isfinite(number):
        raise DaytonaExecutionError(f"worker {name} must be finite")
    return number


def _validate_observations(value: Any, steps: int) -> None:
    if not isinstance(value, list) or len(value) != steps:
        raise DaytonaExecutionError(
            "worker observations must contain one entry per step"
        )
    observation_dimension: int | None = None
    for observation in value:
        if not isinstance(observation, list) or not observation:
            raise DaytonaExecutionError(
                "worker observations must be non-empty numeric lists"
            )
        if observation_dimension is None:
            observation_dimension = len(observation)
        elif len(observation) != observation_dimension:
            raise DaytonaExecutionError(
                "worker observations must share one fixed dimension"
            )
        for item in observation:
            _finite_number(item, "observation")


def _integer_value(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    error_type: type[Exception] = DaytonaExecutionError,
) -> int:
    if isinstance(value, bool):
        raise error_type(f"{name} must be an integer")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise error_type(f"{name} must be an integer") from exc
    if minimum is not None and integer < minimum:
        raise error_type(f"{name} must be at least {minimum}")
    return integer


def _console_event(event: Mapping[str, Any]) -> None:
    world = int(event["world"])
    state = str(event["state"])
    sandbox_id = event.get("sandbox_id") or "-"
    suffix = ""
    if "reward" in event:
        suffix = f" reward={float(event['reward']):.3f}"
    if "error" in event:
        suffix = f" error={event['error']}"
    print(f"WORLD {world:02d} {state:<16} {sandbox_id}{suffix}", flush=True)


def generation_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize an empty generation")
    rewards = [_finite_number(result["reward"], "reward") for result in results]
    best_index = max(range(len(results)), key=rewards.__getitem__)
    best = results[best_index]
    return {
        "worlds": len(results),
        "successful": sum(bool(result["success"]) for result in results),
        "average_reward": sum(rewards) / len(rewards),
        "best_reward": rewards[best_index],
        "best_sandbox": best["sandbox_id"],
        "total_trajectory_points": sum(
            len(result["trajectory"]) for result in results
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parallel Gravity Gauntlet episodes in Daytona"
    )
    parser.add_argument("--worlds", type=int, default=DEFAULT_WORLDS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--policy-version", type=int, default=0)
    parser.add_argument("--policy-weights-file", type=Path)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--snapshot-name", default=DEFAULT_SNAPSHOT_NAME)
    parser.add_argument("--keep-sandboxes", action="store_true")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the exact --output file if it already exists",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> int:
    if args.worlds <= 0:
        raise ValueError("--worlds must be positive")
    if args.output is not None and args.output.exists() and not args.overwrite:
        raise ValueError(
            f"refusing to overwrite existing Daytona result: {args.output}; "
            "choose a new path or pass --overwrite"
        )
    policy_weights = None
    if args.policy_weights_file is not None:
        policy_weights = args.policy_weights_file.read_text(encoding="utf-8").strip()
    seeds = [args.base_seed + index for index in range(args.worlds)]
    started_at = time.perf_counter()
    results = await run_generation(
        policy_version=args.policy_version,
        policy_weights=policy_weights,
        seeds=seeds,
        max_steps=args.max_steps,
        snapshot_name=args.snapshot_name,
        event_callback=_console_event,
        keep_sandboxes=args.keep_sandboxes,
    )
    wall_clock_seconds = time.perf_counter() - started_at
    summary = generation_summary(results)
    summary.update(
        {
            "concurrent": len(results) > 1,
            "wall_clock_seconds": wall_clock_seconds,
            "seeds": [int(result["seed"]) for result in results],
            "sandbox_ids": [str(result["sandbox_id"]) for result in results],
            "cleanup": (
                "retained_by_request"
                if args.keep_sandboxes
                else "explicit_delete_confirmed"
            ),
        }
    )

    print("\nDAYTONA GENERATION COMPLETE")
    print(f"Worlds: {summary['worlds']}")
    print(f"Successful: {summary['successful']}")
    print(f"Average reward: {summary['average_reward']:.3f}")
    print(f"Best reward: {summary['best_reward']:.3f}")
    print(f"Best sandbox: {summary['best_sandbox']}")
    print(f"Total trajectory points: {summary['total_trajectory_points']}")
    print(f"Concurrent: {'yes' if summary['concurrent'] else 'single world'}")
    print(f"Wall-clock seconds: {wall_clock_seconds:.3f}")
    print("Seeds: " + ", ".join(str(seed) for seed in summary["seeds"]))
    print("Sandbox IDs: " + ", ".join(summary["sandbox_ids"]))
    print(f"Cleanup: {summary['cleanup']}")
    if args.keep_sandboxes:
        kept = ", ".join(result["sandbox_id"] for result in results)
        print(f"Kept real sandboxes: {kept}")

    machine_result = {
        "summary": summary,
        "results": results,
        # Agent 2's visual loader accepts a top-level ``rollouts`` list. Keep
        # ``results`` as the canonical Daytona envelope and expose the exact
        # same real dictionaries without regenerating any trajectory data.
        "rollouts": results,
    }
    encoded = json.dumps(machine_result, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"JSON saved: {args.output}")
    else:
        print(f"DAYTONA_RESULTS_JSON={encoded}")
    return 0


def main() -> None:
    args = _parse_args()
    try:
        raise SystemExit(asyncio.run(_async_main(args)))
    except (
        DaytonaConfigurationError,
        DaytonaExecutionError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"DAYTONA GENERATION FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
