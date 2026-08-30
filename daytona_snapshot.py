"""Build and verify the reusable Gravity Gauntlet Daytona worker snapshot."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams
except ImportError:  # A clear error is raised through _require_daytona().
    AsyncDaytona = None  # type: ignore[assignment]
    CreateSandboxFromSnapshotParams = None  # type: ignore[assignment]

from daytona_orchestrator import (
    DEFAULT_SNAPSHOT_NAME,
    REMOTE_JOB_PATH,
    REMOTE_RESULT_PATH,
    REMOTE_WORK_DIR,
    SANDBOX_CREATE_TIMEOUT,
    SANDBOX_DELETE_TIMEOUT,
    DaytonaConfigurationError,
    DaytonaExecutionError,
    _execute_worker_job,
    _real_sandbox_id,
    _require_daytona,
    _upload_json,
    _validate_worker_result,
)


SNAPSHOT_TIMEOUT = 300
SNAPSHOT_ACTIVATION_TIMEOUT = 120.0
SNAPSHOT_POLL_INTERVAL = 1.0
INSTALL_TIMEOUT = 600
SMOKE_SEED = 18_473
SMOKE_STEPS = 20
BUILDER_TTL_MINUTES = 30
VERIFIER_TTL_MINUTES = 10
RUNTIME_MANIFEST_NAME = "daytona_runtime_manifest.json"
RUNTIME_MANIFEST_PATH = f"{REMOTE_WORK_DIR}/{RUNTIME_MANIFEST_NAME}"
WORKER_REQUIREMENTS_FILE = "requirements-daytona.txt"
RUNTIME_FILES = (
    "gravity_env.py",
    "rollout_worker.py",
    "rl_policy.py",
    "daytona_worker_entry.py",
    WORKER_REQUIREMENTS_FILE,
)


class DaytonaSnapshotError(DaytonaExecutionError):
    """Raised when the worker snapshot cannot be built or verified safely."""


async def ensure_worker_snapshot(
    snapshot_name: str = DEFAULT_SNAPSHOT_NAME,
) -> dict[str, Any]:
    """Reuse a valid snapshot or build, smoke-test, snapshot, and verify one."""

    _require_daytona()
    if not isinstance(snapshot_name, str) or not snapshot_name.strip():
        raise ValueError("snapshot_name cannot be empty")
    _require_snapshot_sdk_types()
    runtime_files, manifest_bytes, manifest = _capture_runtime_bundle()

    async with AsyncDaytona() as daytona:  # type: ignore[misc,operator]
        existing = await _find_snapshot(daytona, snapshot_name)
        if existing is not None:
            existing = await _ensure_snapshot_active(daytona, existing)
            verification = await _verify_snapshot(
                daytona,
                snapshot_name,
                manifest_bytes,
            )
            return {
                "snapshot_name": snapshot_name,
                "status": "reused",
                "snapshot_id": getattr(existing, "id", None),
                "snapshot_state": _state_value(existing),
                "runtime_bundle_sha256": manifest["bundle_sha256"],
                "builder_sandbox_id": None,
                "verification": verification,
            }

        try:
            builder = await daytona.create(
                CreateSandboxFromSnapshotParams(
                    language="python",
                    labels={
                        "application": "gravity-gauntlet",
                        "purpose": "snapshot-builder",
                    },
                    auto_delete_interval=60,
                    ttl_minutes=BUILDER_TTL_MINUTES,
                ),
                timeout=SANDBOX_CREATE_TIMEOUT,
            )
        except Exception as exc:
            raise DaytonaSnapshotError(
                "could not create the Daytona snapshot-builder sandbox"
            ) from exc
        builder_id = "<unavailable>"
        snapshot_created = False
        builder_result: dict[str, Any] | None = None
        build_failure: Exception | None = None
        cleanup_failure: Exception | None = None
        try:
            builder_id = _real_sandbox_id(builder)
            uploaded = await _upload_runtime_files(
                builder,
                runtime_files,
                manifest_bytes,
            )
            await _install_worker_dependencies(builder)
            builder_smoke = await _smoke_worker(builder)
            await _remove_smoke_artifacts(builder)

            # SDK 0.207.0 cold snapshots capture the prepared filesystem.  Stop
            # the builder after its smoke test, then snapshot that exact state.
            await builder.stop(timeout=SANDBOX_DELETE_TIMEOUT)
            await builder.create_snapshot(snapshot_name, timeout=SNAPSHOT_TIMEOUT)
            snapshot_created = True
            builder_result = {
                "uploaded_files": uploaded,
                "builder_smoke": builder_smoke,
            }
        except Exception as exc:
            build_failure = exc
        finally:
            try:
                await builder.delete(timeout=SANDBOX_DELETE_TIMEOUT, wait=True)
            except Exception as exc:
                cleanup_failure = exc

        if build_failure is not None:
            message = (
                f"snapshot build failed in builder sandbox {builder_id}: "
                f"{type(build_failure).__name__}: {build_failure}"
            )
            if cleanup_failure is not None:
                message += (
                    f"; builder cleanup also failed: "
                    f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                )
            raise DaytonaSnapshotError(message) from build_failure
        if cleanup_failure is not None:
            suffix = " after the snapshot may have been created" if snapshot_created else ""
            raise DaytonaSnapshotError(
                f"builder sandbox {builder_id} could not be deleted{suffix}; "
                "retry to verify/reuse the snapshot before creating another"
            ) from cleanup_failure
        if not snapshot_created or builder_result is None:
            raise DaytonaSnapshotError("snapshot builder completed without a snapshot")

        created = await _find_snapshot(daytona, snapshot_name)
        if created is None:
            raise DaytonaSnapshotError(
                f"snapshot {snapshot_name!r} was not visible after creation"
            )
        created = await _ensure_snapshot_active(daytona, created)
        verification = await _verify_snapshot(
            daytona,
            snapshot_name,
            manifest_bytes,
        )
        return {
            "snapshot_name": snapshot_name,
            "status": "created",
            "snapshot_id": getattr(created, "id", None),
            "snapshot_state": _state_value(created),
            "runtime_bundle_sha256": manifest["bundle_sha256"],
            "builder_sandbox_id": builder_id,
            **builder_result,
            "verification": verification,
        }


def _capture_runtime_bundle() -> tuple[dict[str, bytes], bytes, dict[str, Any]]:
    repo_root = Path(__file__).resolve().parent
    missing = [name for name in RUNTIME_FILES if not (repo_root / name).is_file()]
    if missing:
        raise DaytonaSnapshotError(
            "canonical worker files are missing: " + ", ".join(missing)
        )

    # Capture each file once. Hashing and upload then use exactly the same
    # bytes even if another agent changes a read-only runtime file mid-build.
    runtime_files = {
        name: (repo_root / name).read_bytes() for name in RUNTIME_FILES
    }
    file_hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in sorted(runtime_files.items())
    }
    bundle_hash = hashlib.sha256()
    for name, content in sorted(runtime_files.items()):
        bundle_hash.update(name.encode("utf-8"))
        bundle_hash.update(b"\0")
        bundle_hash.update(content)
        bundle_hash.update(b"\0")
    manifest = {
        "format": "gravity-gauntlet-daytona-runtime",
        "version": 1,
        "files": file_hashes,
        "bundle_sha256": bundle_hash.hexdigest(),
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    return runtime_files, manifest_bytes, manifest


async def _upload_runtime_files(
    sandbox: Any,
    runtime_files: dict[str, bytes],
    manifest_bytes: bytes,
) -> list[str]:
    await sandbox.fs.create_folder(REMOTE_WORK_DIR, "755")
    for name, content in runtime_files.items():
        await sandbox.fs.upload_file(
            content,
            f"{REMOTE_WORK_DIR}/{name}",
        )
    await sandbox.fs.upload_file(manifest_bytes, RUNTIME_MANIFEST_PATH)
    return [*runtime_files, RUNTIME_MANIFEST_NAME]


async def _install_worker_dependencies(sandbox: Any) -> None:
    response = await sandbox.process.exec(
        f"python3 -m pip install --no-cache-dir -r {WORKER_REQUIREMENTS_FILE}",
        cwd=REMOTE_WORK_DIR,
        timeout=INSTALL_TIMEOUT,
    )
    if getattr(response, "exit_code", None) != 0:
        output = str(getattr(response, "result", ""))[-1000:]
        raise DaytonaSnapshotError(
            f"worker dependency installation failed: {output}"
        )


async def _smoke_worker(sandbox: Any) -> dict[str, Any]:
    sandbox_id = _real_sandbox_id(sandbox)
    job = {
        "sandbox_id": sandbox_id,
        "seed": SMOKE_SEED,
        "policy_version": 0,
        "policy_weights": None,
        "max_steps": SMOKE_STEPS,
    }
    await _upload_json(sandbox, REMOTE_JOB_PATH, job)
    result = await _execute_worker_job(sandbox)
    _validate_worker_result(
        result,
        sandbox_id=sandbox_id,
        seed=SMOKE_SEED,
        policy_version=0,
    )
    return {
        "sandbox_id": sandbox_id,
        "seed": result["seed"],
        "steps": result["steps"],
        "termination": result["termination"],
        "trajectory_points": len(result["trajectory"]),
    }


async def _remove_smoke_artifacts(sandbox: Any) -> None:
    await sandbox.fs.delete_file(REMOTE_JOB_PATH)
    await sandbox.fs.delete_file(REMOTE_RESULT_PATH)


async def _verify_snapshot(
    daytona: Any,
    snapshot_name: str,
    expected_manifest: bytes,
) -> dict[str, Any]:
    try:
        verifier = await daytona.create(
            CreateSandboxFromSnapshotParams(
                snapshot=snapshot_name,
                language="python",
                labels={
                    "application": "gravity-gauntlet",
                    "purpose": "snapshot-verification",
                },
                auto_delete_interval=30,
                ttl_minutes=VERIFIER_TTL_MINUTES,
            ),
            timeout=SANDBOX_CREATE_TIMEOUT,
        )
    except Exception as exc:
        raise DaytonaSnapshotError(
            f"could not create a verifier from snapshot {snapshot_name!r}"
        ) from exc
    verifier_id = "<unavailable>"
    verification: dict[str, Any] | None = None
    verification_failure: Exception | None = None
    cleanup_failure: Exception | None = None
    try:
        verifier_id = _real_sandbox_id(verifier)
        await _verify_runtime_manifest(verifier, expected_manifest)
        verification = await _smoke_worker(verifier)
    except Exception as exc:
        verification_failure = exc
    finally:
        try:
            await verifier.delete(timeout=SANDBOX_DELETE_TIMEOUT, wait=True)
        except Exception as exc:
            cleanup_failure = exc

    if verification_failure is not None:
        message = (
            f"snapshot {snapshot_name!r} failed verification in {verifier_id}: "
            f"{type(verification_failure).__name__}: {verification_failure}"
        )
        if cleanup_failure is not None:
            message += (
                f"; verifier cleanup also failed: "
                f"{type(cleanup_failure).__name__}: {cleanup_failure}"
            )
        raise DaytonaSnapshotError(message) from verification_failure
    if cleanup_failure is not None:
        raise DaytonaSnapshotError(
            f"snapshot verifier sandbox {verifier_id} could not be deleted"
        ) from cleanup_failure
    if verification is None:
        raise DaytonaSnapshotError(
            f"snapshot {snapshot_name!r} produced no verification result"
        )
    return verification


async def _verify_runtime_manifest(
    sandbox: Any,
    expected_manifest: bytes,
) -> None:
    try:
        actual_manifest = await sandbox.fs.download_file(RUNTIME_MANIFEST_PATH)
    except Exception as exc:
        raise DaytonaSnapshotError(
            "snapshot has no readable runtime manifest and may be stale"
        ) from exc
    if not isinstance(actual_manifest, (bytes, bytearray)):
        raise DaytonaSnapshotError("snapshot runtime manifest download was not bytes")
    if bytes(actual_manifest) != expected_manifest:
        raise DaytonaSnapshotError(
            "snapshot runtime manifest does not match the canonical worker; "
            "use a new snapshot name or explicitly manage the existing snapshot"
        )


async def _find_snapshot(daytona: Any, name: str) -> Any | None:
    page = 1
    while True:
        response = await daytona.snapshot.list(page=page, limit=100)
        for snapshot in response.items:
            if getattr(snapshot, "name", None) == name:
                return snapshot
        total_pages = max(1, int(getattr(response, "total_pages", 1)))
        if page >= total_pages:
            return None
        page += 1


async def _ensure_snapshot_active(daytona: Any, snapshot: Any) -> Any:
    state = _state_value(snapshot)
    error_reason = getattr(snapshot, "error_reason", None)
    if state in {"error", "build_failed"}:
        raise DaytonaSnapshotError(
            f"existing snapshot {snapshot.name!r} is invalid "
            f"(state={state}, reason={error_reason or 'not provided'})"
        )
    if state == "inactive":
        try:
            snapshot = await daytona.snapshot.activate(snapshot)
        except Exception as exc:
            raise DaytonaSnapshotError(
                f"snapshot {snapshot.name!r} could not be activated"
            ) from exc
        state = _state_value(snapshot)
    if state == "active":
        return snapshot
    if state in {"building", "pending", "pulling", "snapshotting"}:
        return await _wait_for_active_snapshot(daytona, snapshot)
    if state == "removing":
        raise DaytonaSnapshotError(
            f"snapshot {snapshot.name!r} is being removed; retry later"
        )
    raise DaytonaSnapshotError(
        f"snapshot {snapshot.name!r} has unsupported state {state!r}"
    )


async def _wait_for_active_snapshot(daytona: Any, snapshot: Any) -> Any:
    identifier = getattr(snapshot, "id", None) or getattr(snapshot, "name", None)
    if not isinstance(identifier, str) or not identifier:
        raise DaytonaSnapshotError("snapshot has no id or name for state polling")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + SNAPSHOT_ACTIVATION_TIMEOUT
    while True:
        state = _state_value(snapshot)
        if state == "active":
            return snapshot
        if state in {"error", "build_failed"}:
            error_reason = getattr(snapshot, "error_reason", None)
            raise DaytonaSnapshotError(
                f"snapshot {getattr(snapshot, 'name', identifier)!r} failed while "
                f"becoming active (state={state}, "
                f"reason={error_reason or 'not provided'})"
            )
        if state == "removing":
            raise DaytonaSnapshotError(
                f"snapshot {getattr(snapshot, 'name', identifier)!r} was removed "
                "while becoming active"
            )
        if state not in {"building", "pending", "pulling", "snapshotting"}:
            raise DaytonaSnapshotError(
                f"snapshot {getattr(snapshot, 'name', identifier)!r} has "
                f"unsupported state {state!r} while becoming active"
            )

        remaining = deadline - loop.time()
        if remaining <= 0:
            raise DaytonaSnapshotError(
                f"snapshot {getattr(snapshot, 'name', identifier)!r} did not "
                f"become active within {SNAPSHOT_ACTIVATION_TIMEOUT:.0f} seconds "
                f"(last state={state})"
            )
        await asyncio.sleep(min(SNAPSHOT_POLL_INTERVAL, remaining))
        try:
            snapshot = await daytona.snapshot.get(identifier)
        except Exception as exc:
            raise DaytonaSnapshotError(
                f"snapshot {getattr(snapshot, 'name', identifier)!r} state "
                "could not be refreshed"
            ) from exc


def _state_value(snapshot: Any) -> str:
    state = getattr(snapshot, "state", "unknown")
    value = getattr(state, "value", state)
    return str(value).lower()


def _require_snapshot_sdk_types() -> None:
    if AsyncDaytona is None or CreateSandboxFromSnapshotParams is None:
        raise DaytonaSnapshotError(
            "Daytona SDK is unavailable on the host; install daytona==0.207.0"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify the Gravity Gauntlet Daytona snapshot"
    )
    parser.add_argument("--snapshot-name", default=DEFAULT_SNAPSHOT_NAME)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = asyncio.run(ensure_worker_snapshot(args.snapshot_name))
    except (DaytonaConfigurationError, DaytonaExecutionError, ValueError) as exc:
        raise SystemExit(f"DAYTONA SNAPSHOT FAILED: {exc}") from exc
    except Exception as exc:
        raise SystemExit(
            f"DAYTONA SNAPSHOT FAILED: {type(exc).__name__}: {exc}"
        ) from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
