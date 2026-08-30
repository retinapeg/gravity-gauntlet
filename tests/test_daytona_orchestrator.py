"""Offline contract tests for Daytona orchestration and snapshot handling.

The fakes in this module deliberately use the same upload/execute/download
boundary as Daytona.  No test creates a real client or makes a network call.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import daytona_orchestrator as orchestrator
import daytona_snapshot as snapshot
import daytona_worker_entry as worker_entry


class FakeCreateParams:
    """Small stand-in for the SDK's create-parameter model."""

    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)


class AsyncArrivalBarrier:
    """Release workers only after every expected world has started execution."""

    def __init__(self, parties: int) -> None:
        self.parties = parties
        self.arrivals: list[str] = []
        self._released = asyncio.Event()

    async def wait(self, sandbox_id: str) -> None:
        self.arrivals.append(sandbox_id)
        if len(self.arrivals) == self.parties:
            self._released.set()
        await asyncio.wait_for(self._released.wait(), timeout=2.0)


def worker_result(job: dict[str, object], sandbox_id: str) -> dict[str, object]:
    rewards = [0.25, -0.10]
    return {
        "sandbox_id": sandbox_id,
        "seed": job["seed"],
        "policy_version": job["policy_version"],
        "reward": sum(rewards),
        "success": False,
        "termination": "timeout",
        "steps": 2,
        "physics_steps": 2,
        "trajectory": [
            {
                "step": step,
                "x": 100.0 + step,
                "y": 400.0 - step,
                "vx": 1.0,
                "vy": -1.0,
                "reward": 0.0,
                "clearance": 50.0,
                "status": "running" if step < 2 else "timeout",
            }
            for step in range(3)
        ],
        "observations": [[0.0, 0.0], [0.1, 0.2]],
        "actions": [0, 1],
        "rewards": rewards,
    }


class FakeFilesystem:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self.sandbox = sandbox

    async def create_folder(self, path: str, mode: str) -> None:
        self.sandbox.log.append(("folder", self.sandbox.id, path, mode))

    async def upload_file(self, content: bytes, path: str) -> None:
        data = bytes(content)
        self.sandbox.files[path] = data
        self.sandbox.log.append(("upload", self.sandbox.id, path, data))

    async def download_file(self, path: str) -> bytes:
        self.sandbox.log.append(("download", self.sandbox.id, path))
        if path == snapshot.RUNTIME_MANIFEST_PATH:
            if self.sandbox.manifest is None:
                raise FileNotFoundError(path)
            return self.sandbox.manifest
        if path != orchestrator.REMOTE_RESULT_PATH:
            raise FileNotFoundError(path)

        job = json.loads(
            self.sandbox.files[orchestrator.REMOTE_JOB_PATH].decode("utf-8")
        )
        result: object = worker_result(job, self.sandbox.id)
        if self.sandbox.result_transform is not None:
            result = self.sandbox.result_transform(result)
        if isinstance(result, bytes):
            return result
        return json.dumps(result, sort_keys=True, allow_nan=False).encode("utf-8")

    async def delete_file(self, path: str) -> None:
        self.sandbox.log.append(("delete_file", self.sandbox.id, path))
        self.sandbox.files.pop(path, None)


class FakeProcess:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self.sandbox = sandbox

    async def exec(self, command: str, *, cwd: str, timeout: int) -> object:
        self.sandbox.log.append(
            ("exec", self.sandbox.id, command, cwd, timeout)
        )
        if command == orchestrator.WORKER_COMMAND:
            if self.sandbox.barrier is not None:
                await self.sandbox.barrier.wait(self.sandbox.id)
            if self.sandbox.worker_failure:
                return SimpleNamespace(exit_code=17, result="deliberate worker failure")
        if self.sandbox.install_failure and command.startswith("python3 -m pip"):
            return SimpleNamespace(exit_code=1, result="deliberate install failure")
        return SimpleNamespace(exit_code=0, result="ok")


class FakeSandbox:
    def __init__(
        self,
        sandbox_id: str,
        *,
        log: list[tuple[object, ...]],
        barrier: AsyncArrivalBarrier | None = None,
        worker_failure: bool = False,
        install_failure: bool = False,
        delete_failure: bool = False,
        manifest: bytes | None = None,
        result_transform: object | None = None,
        on_snapshot: object | None = None,
    ) -> None:
        self.id = sandbox_id
        self.log = log
        self.barrier = barrier
        self.worker_failure = worker_failure
        self.install_failure = install_failure
        self.delete_failure = delete_failure
        self.manifest = manifest
        self.result_transform = result_transform
        self.on_snapshot = on_snapshot
        self.files: dict[str, bytes] = {}
        self.delete_attempts = 0
        self.deleted = 0
        self.stopped = 0
        self.created_snapshots: list[str] = []
        self.fs = FakeFilesystem(self)
        self.process = FakeProcess(self)

    async def delete(self, *, timeout: int, wait: bool) -> None:
        self.delete_attempts += 1
        self.log.append(("delete", self.id, timeout, wait))
        if self.delete_failure:
            raise RuntimeError("deliberate cleanup failure")
        self.deleted += 1

    async def stop(self, *, timeout: int) -> None:
        self.stopped += 1
        self.log.append(("stop", self.id, timeout))

    async def create_snapshot(self, name: str, *, timeout: int) -> None:
        self.created_snapshots.append(name)
        self.log.append(("create_snapshot", self.id, name, timeout))
        if self.on_snapshot is not None:
            self.on_snapshot(name)


class FakeSnapshotService:
    def __init__(
        self,
        items: list[object] | None = None,
        *,
        log: list[tuple[object, ...]],
    ) -> None:
        self.items = list(items or [])
        self.log = log
        self.delete_calls: list[object] = []

    async def list(self, *, page: int, limit: int) -> object:
        self.log.append(("snapshot.list", page, limit))
        return SimpleNamespace(items=list(self.items), total_pages=1)

    async def activate(self, item: object) -> object:
        self.log.append(("snapshot.activate", getattr(item, "name", None)))
        item.state = SimpleNamespace(value="active")
        return item

    async def delete(self, item: object) -> None:
        self.delete_calls.append(item)
        self.log.append(("snapshot.delete", getattr(item, "name", None)))


class FakeDaytona:
    def __init__(
        self,
        sandbox_factory: object,
        *,
        log: list[tuple[object, ...]],
        snapshot_service: FakeSnapshotService | None = None,
    ) -> None:
        self.sandbox_factory = sandbox_factory
        self.log = log
        self.snapshot = snapshot_service or FakeSnapshotService(log=log)
        self.sandboxes: list[FakeSandbox] = []
        self.create_calls: list[FakeCreateParams] = []
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> "FakeDaytona":
        self.entered += 1
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.exited += 1

    async def create(self, params: FakeCreateParams, *, timeout: int) -> FakeSandbox:
        purpose = getattr(params, "labels", {}).get("purpose", "unknown")
        self.log.append(("create", purpose, timeout))
        self.create_calls.append(params)
        sandbox = self.sandbox_factory(params)
        self.sandboxes.append(sandbox)
        await asyncio.sleep(0)
        return sandbox


def active_snapshot(name: str, snapshot_id: str = "snapshot-1") -> object:
    return SimpleNamespace(
        id=snapshot_id,
        name=name,
        state=SimpleNamespace(value="active"),
        error_reason=None,
    )


@contextlib.contextmanager
def patched_orchestrator(client: FakeDaytona):
    """Install an offline SDK surface while retaining production validation."""

    with (
        mock.patch.dict(os.environ, {"DAYTONA_API_KEY": "offline-test-key"}),
        mock.patch.object(orchestrator, "AsyncDaytona", lambda: client),
        mock.patch.object(
            orchestrator,
            "CreateSandboxFromSnapshotParams",
            FakeCreateParams,
        ),
    ):
        yield


@contextlib.contextmanager
def patched_snapshot(client: FakeDaytona):
    with (
        patched_orchestrator(client),
        mock.patch.object(snapshot, "AsyncDaytona", lambda: client),
        mock.patch.object(
            snapshot,
            "CreateSandboxFromSnapshotParams",
            FakeCreateParams,
        ),
    ):
        yield


class DaytonaGenerationTests(unittest.IsolatedAsyncioTestCase):
    def test_missing_python_ca_uses_certifi_without_disabling_tls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "cacert.pem"
            bundle.write_text("test CA bundle", encoding="utf-8")
            with (
                mock.patch.dict(os.environ, {}, clear=False),
                mock.patch.object(
                    orchestrator.ssl,
                    "get_default_verify_paths",
                    return_value=SimpleNamespace(
                        cafile=None,
                        capath=None,
                        openssl_cafile_env="SSL_CERT_FILE",
                        openssl_capath_env="SSL_CERT_DIR",
                    ),
                ),
                mock.patch("certifi.where", return_value=str(bundle)),
            ):
                os.environ.pop("SSL_CERT_FILE", None)
                os.environ.pop("SSL_CERT_DIR", None)

                selected = orchestrator._configure_tls_ca_bundle()

                self.assertEqual(selected, str(bundle))
                self.assertEqual(os.environ["SSL_CERT_FILE"], str(bundle))

    def test_explicit_ca_configuration_is_preserved(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"SSL_CERT_FILE": "/private/company-ca.pem"},
                clear=False,
            ),
            mock.patch.object(
                orchestrator.ssl,
                "get_default_verify_paths",
                return_value=SimpleNamespace(
                    cafile=None,
                    capath=None,
                    openssl_cafile_env="SSL_CERT_FILE",
                    openssl_capath_env="SSL_CERT_DIR",
                ),
            ),
            mock.patch(
                "certifi.where",
                side_effect=AssertionError("explicit CA must win"),
            ),
        ):
            selected = orchestrator._configure_tls_ca_bundle()

            self.assertIsNone(selected)
            self.assertEqual(
                os.environ["SSL_CERT_FILE"],
                "/private/company-ca.pem",
            )

    async def test_eight_worlds_run_concurrently_with_exact_transport_and_events(
        self,
    ) -> None:
        log: list[tuple[object, ...]] = []
        barrier = AsyncArrivalBarrier(8)

        def factory(params: FakeCreateParams) -> FakeSandbox:
            world = int(params.labels["world"])
            return FakeSandbox(f"sandbox-{world}", log=log, barrier=barrier)

        client = FakeDaytona(factory, log=log)
        seeds = [41_000 + index for index in range(8)]
        weights = "frozen-policy-weights"
        events: list[dict[str, object]] = []

        with patched_orchestrator(client):
            results = await orchestrator.run_generation(
                policy_version=6,
                policy_weights=weights,
                seeds=seeds,
                max_steps=321,
                snapshot_name="test-snapshot",
                event_callback=events.append,
            )

        self.assertCountEqual(
            barrier.arrivals,
            [f"sandbox-{world}" for world in range(1, 9)],
        )
        self.assertEqual([result["seed"] for result in results], seeds)
        self.assertEqual(len(set(seeds)), 8)
        self.assertEqual(client.entered, 1)
        self.assertEqual(client.exited, 1)

        jobs: list[dict[str, object]] = []
        for world, sandbox in enumerate(client.sandboxes, start=1):
            jobs.append(
                json.loads(
                    sandbox.files[orchestrator.REMOTE_JOB_PATH].decode("utf-8")
                )
            )
            uploads = [
                item
                for item in log
                if item[:3]
                == ("upload", sandbox.id, orchestrator.REMOTE_JOB_PATH)
            ]
            downloads = [
                item
                for item in log
                if item
                == ("download", sandbox.id, orchestrator.REMOTE_RESULT_PATH)
            ]
            self.assertEqual(len(uploads), 1)
            self.assertEqual(len(downloads), 1)
            self.assertEqual(sandbox.deleted, 1)

            reward = sum([0.25, -0.10])
            expected_events = [
                {
                    "world": world,
                    "seed": seeds[world - 1],
                    "state": "CREATING",
                    "sandbox_id": None,
                },
                {
                    "world": world,
                    "seed": seeds[world - 1],
                    "state": "LIVE",
                    "sandbox_id": sandbox.id,
                },
                {
                    "world": world,
                    "seed": seeds[world - 1],
                    "state": "RUNNING",
                    "sandbox_id": sandbox.id,
                },
                {
                    "world": world,
                    "seed": seeds[world - 1],
                    "state": "TIMEOUT",
                    "sandbox_id": sandbox.id,
                    "reward": reward,
                    "termination": "timeout",
                },
                {
                    "world": world,
                    "seed": seeds[world - 1],
                    "state": "RESULT_COLLECTED",
                    "sandbox_id": sandbox.id,
                    "reward": reward,
                },
            ]
            self.assertEqual(
                [event for event in events if event["world"] == world],
                expected_events,
            )

        self.assertEqual([job["seed"] for job in jobs], seeds)
        self.assertEqual({job["policy_version"] for job in jobs}, {6})
        self.assertEqual({job["policy_weights"] for job in jobs}, {weights})
        self.assertEqual({job["max_steps"] for job in jobs}, {321})
        self.assertEqual(
            [job["sandbox_id"] for job in jobs],
            [f"sandbox-{world}" for world in range(1, 9)],
        )

    async def test_async_callback_is_awaited_and_receives_full_lifecycle(self) -> None:
        log: list[tuple[object, ...]] = []
        client = FakeDaytona(
            lambda params: FakeSandbox("sandbox-async", log=log),
            log=log,
        )
        events: list[dict[str, object]] = []

        async def callback(event: dict[str, object]) -> None:
            await asyncio.sleep(0)
            events.append(event)

        with patched_orchestrator(client):
            await orchestrator.run_generation(
                policy_version=0,
                policy_weights=None,
                seeds=[77],
                event_callback=callback,
            )

        self.assertEqual(
            [event["state"] for event in events],
            ["CREATING", "LIVE", "RUNNING", "TIMEOUT", "RESULT_COLLECTED"],
        )

    async def test_one_world_failure_keeps_seven_results_and_cleans_every_sandbox(
        self,
    ) -> None:
        log: list[tuple[object, ...]] = []
        barrier = AsyncArrivalBarrier(8)

        def factory(params: FakeCreateParams) -> FakeSandbox:
            world = int(params.labels["world"])
            return FakeSandbox(
                f"sandbox-{world}",
                log=log,
                barrier=barrier,
                worker_failure=world == 4,
            )

        client = FakeDaytona(factory, log=log)
        seeds = [52_000 + index for index in range(8)]
        events: list[dict[str, object]] = []

        with patched_orchestrator(client):
            with self.assertRaises(orchestrator.DaytonaGenerationError) as raised:
                await orchestrator.run_generation(
                    policy_version=3,
                    policy_weights="one-frozen-policy",
                    seeds=seeds,
                    event_callback=events.append,
                )

        error = raised.exception
        self.assertEqual(len(error.failures), 1)
        self.assertEqual(error.failures[0]["world"], 4)
        self.assertEqual(error.failures[0]["seed"], seeds[3])
        self.assertIn("worker exited with code 17", error.failures[0]["error"])
        self.assertEqual(len(error.partial_results), 7)
        self.assertNotIn(seeds[3], [result["seed"] for result in error.partial_results])
        self.assertEqual(len(barrier.arrivals), 8)
        self.assertTrue(all(sandbox.deleted == 1 for sandbox in client.sandboxes))
        self.assertEqual(
            [event["state"] for event in events if event["world"] == 4],
            ["CREATING", "LIVE", "RUNNING", "ERROR"],
        )

    async def test_keep_sandboxes_skips_deletion(self) -> None:
        log: list[tuple[object, ...]] = []
        client = FakeDaytona(
            lambda params: FakeSandbox(
                f"kept-{params.labels['world']}",
                log=log,
            ),
            log=log,
        )

        with patched_orchestrator(client):
            results = await orchestrator.run_generation(
                policy_version=0,
                policy_weights=None,
                seeds=[901, 902],
                keep_sandboxes=True,
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(sandbox.delete_attempts == 0 for sandbox in client.sandboxes))
        self.assertFalse(any(item[0] == "delete" for item in log))

    async def test_worker_and_cleanup_failures_are_both_reported(self) -> None:
        log: list[tuple[object, ...]] = []
        sandbox = FakeSandbox(
            "failed-and-leaked",
            log=log,
            worker_failure=True,
            delete_failure=True,
        )
        client = FakeDaytona(lambda params: sandbox, log=log)

        with patched_orchestrator(client):
            with self.assertRaisesRegex(
                orchestrator.DaytonaExecutionError,
                "worker exited with code 17.*cleanup also failed",
            ):
                await orchestrator.run_world(
                    client,
                    snapshot_name="test-snapshot",
                    world_index=1,
                    seed=123,
                    policy_version=0,
                    policy_weights=None,
                    max_steps=20,
                )

        self.assertEqual(sandbox.delete_attempts, 1)
        self.assertEqual(sandbox.deleted, 0)

    async def test_missing_credential_rejects_before_client_creation(self) -> None:
        client_creations = 0

        def forbidden_client() -> object:
            nonlocal client_creations
            client_creations += 1
            raise AssertionError("client must not be constructed")

        with (
            mock.patch.dict(os.environ, {"DAYTONA_API_KEY": ""}),
            mock.patch.object(orchestrator, "AsyncDaytona", forbidden_client),
            mock.patch.object(
                orchestrator,
                "CreateSandboxFromSnapshotParams",
                FakeCreateParams,
            ),
        ):
            with self.assertRaisesRegex(
                orchestrator.DaytonaConfigurationError,
                "DAYTONA_API_KEY",
            ):
                await orchestrator.run_generation(
                    policy_version=0,
                    policy_weights=None,
                    seeds=[1],
                )

        self.assertEqual(client_creations, 0)

    async def test_wrong_sandbox_id_and_invalid_result_are_rejected_and_cleaned(
        self,
    ) -> None:
        def wrong_id(result: object) -> object:
            return {**result, "sandbox_id": "forged-id"}  # type: ignore[arg-type]

        def missing_actions(result: object) -> object:
            changed = dict(result)  # type: ignore[arg-type]
            changed.pop("actions")
            return changed

        cases = (
            (wrong_id, "sandbox_id does not match"),
            (missing_actions, "missing fields: actions"),
        )
        for index, (transform, message) in enumerate(cases, start=1):
            with self.subTest(case=message):
                log: list[tuple[object, ...]] = []
                sandbox = FakeSandbox(
                    f"invalid-{index}",
                    log=log,
                    result_transform=transform,
                )
                client = FakeDaytona(lambda params: sandbox, log=log)
                with patched_orchestrator(client):
                    with self.assertRaisesRegex(
                        orchestrator.DaytonaExecutionError,
                        message,
                    ):
                        await orchestrator.run_world(
                            client,
                            snapshot_name="test-snapshot",
                            world_index=index,
                            seed=100 + index,
                            policy_version=0,
                            policy_weights=None,
                            max_steps=20,
                        )
                self.assertEqual(sandbox.deleted, 1)
                self.assertIn(
                    ("download", sandbox.id, orchestrator.REMOTE_RESULT_PATH),
                    log,
                )


class DaytonaWorkerBridgeTests(unittest.TestCase):
    def test_bridge_preserves_categorical_actions_and_rich_trajectory(self) -> None:
        raw_result = {
            "sandbox_id": None,
            "seed": 18473,
            "policy_version": 0,
            "reward": 0.5,
            "success": True,
            "termination": "portal",
            "steps": 1,
            "physics_steps": 1,
            "trajectory": [
                {
                    "step": 0,
                    "x": 100.0,
                    "y": 400.0,
                    "vx": 2.0,
                    "vy": -1.0,
                    "reward": 0.5,
                    "clearance": 25.0,
                    "status": "portal",
                },
                {
                    "step": 1,
                    "x": 101.0,
                    "y": 399.0,
                    "vx": 2.0,
                    "vy": -1.0,
                    "reward": 0.0,
                    "clearance": 24.0,
                    "status": "portal",
                },
            ],
            "observations": [[0.1, 0.2]],
            "actions": [7],
            "action_vectors": [[-1.0, 0.0]],
            "rewards": [0.5],
            "useful_extra": {"preserved": True},
        }
        job = {
            "sandbox_id": "real-sandbox-id",
            "seed": 18473,
            "policy_version": 0,
            "policy_weights": None,
            "max_steps": 50,
        }

        with mock.patch.object(worker_entry, "execute_job", return_value=raw_result):
            result = worker_entry.execute_daytona_job(job)

        self.assertEqual(result["sandbox_id"], "real-sandbox-id")
        self.assertEqual(result["termination"], "success")
        self.assertEqual(result["worker_termination"], "portal")
        self.assertEqual(result["actions"], [7])
        self.assertEqual(result["action_vectors"], [[-1.0, 0.0]])
        self.assertEqual(result["trajectory"], raw_result["trajectory"])
        self.assertEqual(result["useful_extra"], {"preserved": True})

    def test_bridge_rejects_a_conflicting_worker_sandbox_id(self) -> None:
        raw_result = {
            "sandbox_id": "forged-sandbox-id",
            "seed": 18473,
            "policy_version": 0,
        }
        job = {
            "sandbox_id": "real-sandbox-id",
            "seed": 18473,
            "policy_version": 0,
            "policy_weights": None,
            "max_steps": 50,
        }

        with mock.patch.object(worker_entry, "execute_job", return_value=raw_result):
            with self.assertRaisesRegex(
                worker_entry.DaytonaWorkerContractError,
                "sandbox_id conflicts",
            ):
                worker_entry.execute_daytona_job(job)

    def test_bridge_requires_observations_for_the_trainer(self) -> None:
        raw_result = worker_result(
            {
                "seed": 18473,
                "policy_version": 0,
            },
            "real-sandbox-id",
        )
        del raw_result["observations"]
        job = {
            "sandbox_id": "real-sandbox-id",
            "seed": 18473,
            "policy_version": 0,
            "policy_weights": None,
            "max_steps": 50,
        }

        with mock.patch.object(worker_entry, "execute_job", return_value=raw_result):
            with self.assertRaisesRegex(
                worker_entry.DaytonaWorkerContractError,
                "missing fields: observations",
            ):
                worker_entry.execute_daytona_job(job)


class DaytonaCliArtifactTests(unittest.IsolatedAsyncioTestCase):
    def test_host_validator_requires_aligned_trainer_and_physics_shapes(self) -> None:
        result = worker_result(
            {"seed": 18_473, "policy_version": 0},
            "real-fixture-sandbox",
        )
        orchestrator._validate_worker_result(
            result,
            sandbox_id="real-fixture-sandbox",
            seed=18_473,
            policy_version=0,
        )

        missing_observation = dict(result)
        missing_observation["observations"] = [[0.0, 0.0]]
        with self.assertRaisesRegex(
            orchestrator.DaytonaExecutionError,
            "observations must contain one entry per step",
        ):
            orchestrator._validate_worker_result(
                missing_observation,
                sandbox_id="real-fixture-sandbox",
                seed=18_473,
                policy_version=0,
            )

        wrong_physics_length = dict(result)
        wrong_physics_length["physics_steps"] = 7
        with self.assertRaisesRegex(
            orchestrator.DaytonaExecutionError,
            "initial point plus one point per physics step",
        ):
            orchestrator._validate_worker_result(
                wrong_physics_length,
                sandbox_id="real-fixture-sandbox",
                seed=18_473,
                policy_version=0,
            )

    def test_duplicate_sandbox_ids_are_rejected_before_return(self) -> None:
        first = worker_result(
            {"seed": 18_473, "policy_version": 0},
            "duplicate-sandbox-id",
        )
        second = worker_result(
            {"seed": 18_474, "policy_version": 0},
            "duplicate-sandbox-id",
        )
        with self.assertRaisesRegex(
            orchestrator.DaytonaExecutionError,
            "duplicate sandbox IDs",
        ):
            orchestrator._validate_unique_sandbox_ids([first, second])

    async def test_cli_artifact_is_measured_and_visual_loader_compatible(self) -> None:
        results = [
            worker_result({"seed": seed, "policy_version": 0}, sandbox_id)
            for seed, sandbox_id in (
                (18_473, "real-fixture-sandbox-1"),
                (18_474, "real-fixture-sandbox-2"),
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs" / "daytona_v0_smoke.json"
            args = SimpleNamespace(
                worlds=2,
                base_seed=18_473,
                policy_version=0,
                policy_weights_file=None,
                max_steps=20,
                snapshot_name="gravity-gauntlet-worker-v2",
                keep_sandboxes=False,
                output=output,
                overwrite=False,
            )
            with (
                mock.patch.object(
                    orchestrator,
                    "run_generation",
                    new=mock.AsyncMock(return_value=results),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = await orchestrator._async_main(args)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["results"], results)
        self.assertEqual(payload["rollouts"], results)
        self.assertTrue(payload["summary"]["concurrent"])
        self.assertGreaterEqual(payload["summary"]["wall_clock_seconds"], 0.0)
        self.assertEqual(payload["summary"]["seeds"], [18_473, 18_474])
        self.assertEqual(
            payload["summary"]["sandbox_ids"],
            ["real-fixture-sandbox-1", "real-fixture-sandbox-2"],
        )
        self.assertEqual(
            payload["summary"]["cleanup"],
            "explicit_delete_confirmed",
        )

    async def test_cli_refuses_existing_output_before_daytona(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("preserve me\n", encoding="utf-8")
            args = SimpleNamespace(
                worlds=2,
                base_seed=18_473,
                policy_version=0,
                policy_weights_file=None,
                max_steps=20,
                snapshot_name="gravity-gauntlet-worker-v2",
                keep_sandboxes=False,
                output=output,
                overwrite=False,
            )
            with mock.patch.object(
                orchestrator,
                "run_generation",
                new=mock.AsyncMock(),
            ) as run_generation:
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    await orchestrator._async_main(args)

            self.assertEqual(output.read_text(encoding="utf-8"), "preserve me\n")
            run_generation.assert_not_awaited()


class DaytonaSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_dependency_install_disables_pip_cache(self) -> None:
        log: list[tuple[object, ...]] = []
        sandbox = FakeSandbox("install-command", log=log)

        await snapshot._install_worker_dependencies(sandbox)

        self.assertIn(
            (
                "exec",
                sandbox.id,
                "python3 -m pip install --no-cache-dir "
                f"-r {snapshot.WORKER_REQUIREMENTS_FILE}",
                orchestrator.REMOTE_WORK_DIR,
                snapshot.INSTALL_TIMEOUT,
            ),
            log,
        )

    def test_runtime_bundle_uses_headless_worker_requirements(self) -> None:
        runtime_files, _, _ = snapshot._capture_runtime_bundle()

        self.assertIn(snapshot.WORKER_REQUIREMENTS_FILE, runtime_files)
        self.assertNotIn("requirements.txt", runtime_files)

        requirement_lines = [
            line.strip().lower()
            for line in runtime_files[snapshot.WORKER_REQUIREMENTS_FILE]
            .decode("utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertTrue(any(line.startswith("torch") for line in requirement_lines))
        self.assertIn(
            "--index-url https://download.pytorch.org/whl/cpu",
            requirement_lines,
        )
        self.assertFalse(any("pygame" in line for line in requirement_lines))
        self.assertFalse(any("daytona" in line for line in requirement_lines))

        for module_name in (
            "gravity_env.py",
            "rl_policy.py",
            "rollout_worker.py",
            "daytona_worker_entry.py",
        ):
            tree = ast.parse(runtime_files[module_name], filename=module_name)
            imported_roots = {
                alias.name.partition(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imported_roots.update(
                node.module.partition(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            )
            self.assertNotIn("pygame", imported_roots, module_name)

    async def test_inactive_snapshot_waits_through_activation_states(self) -> None:
        log: list[tuple[object, ...]] = []
        existing = active_snapshot("worker-snapshot", "snapshot-delayed")
        existing.state = SimpleNamespace(value="inactive")

        class DelayedActivationService(FakeSnapshotService):
            def __init__(self) -> None:
                super().__init__([existing], log=log)
                self.refresh_states = ["pulling", "active"]

            async def activate(self, item: object) -> object:
                self.log.append(("snapshot.activate", getattr(item, "name", None)))
                item.state = SimpleNamespace(value="pending")
                return item

            async def get(self, identifier: str) -> object:
                self.log.append(("snapshot.get", identifier))
                existing.state = SimpleNamespace(value=self.refresh_states.pop(0))
                return existing

        service = DelayedActivationService()
        daytona = SimpleNamespace(snapshot=service)
        with mock.patch.object(snapshot, "SNAPSHOT_POLL_INTERVAL", 0.0):
            activated = await snapshot._ensure_snapshot_active(daytona, existing)

        self.assertIs(activated, existing)
        self.assertEqual(snapshot._state_value(activated), "active")
        self.assertEqual(
            [item for item in log if item[0] == "snapshot.get"],
            [
                ("snapshot.get", "snapshot-delayed"),
                ("snapshot.get", "snapshot-delayed"),
            ],
        )

    async def test_matching_manifest_reuses_snapshot_and_verifies_worker(self) -> None:
        log: list[tuple[object, ...]] = []
        _, manifest_bytes, manifest = snapshot._capture_runtime_bundle()
        existing = active_snapshot("worker-snapshot", "snapshot-existing")
        service = FakeSnapshotService([existing], log=log)
        verifier = FakeSandbox(
            "snapshot-verifier",
            log=log,
            manifest=manifest_bytes,
        )
        client = FakeDaytona(lambda params: verifier, log=log, snapshot_service=service)

        with patched_snapshot(client):
            result = await snapshot.ensure_worker_snapshot("worker-snapshot")

        self.assertEqual(result["status"], "reused")
        self.assertEqual(result["snapshot_id"], "snapshot-existing")
        self.assertEqual(result["runtime_bundle_sha256"], manifest["bundle_sha256"])
        self.assertEqual(result["verification"]["sandbox_id"], verifier.id)
        self.assertEqual(len(client.create_calls), 1)
        self.assertEqual(client.create_calls[0].snapshot, "worker-snapshot")
        self.assertEqual(verifier.deleted, 1)
        self.assertEqual(service.delete_calls, [])

    async def test_mismatched_manifest_rejects_without_deleting_existing_snapshot(
        self,
    ) -> None:
        log: list[tuple[object, ...]] = []
        existing = active_snapshot("worker-snapshot", "snapshot-existing")
        service = FakeSnapshotService([existing], log=log)
        verifier = FakeSandbox(
            "snapshot-verifier",
            log=log,
            manifest=b'{"bundle_sha256":"stale"}\n',
        )
        client = FakeDaytona(lambda params: verifier, log=log, snapshot_service=service)

        with patched_snapshot(client):
            with self.assertRaisesRegex(
                snapshot.DaytonaSnapshotError,
                "manifest does not match",
            ):
                await snapshot.ensure_worker_snapshot("worker-snapshot")

        self.assertEqual(service.delete_calls, [])
        self.assertFalse(any(item[0] == "snapshot.delete" for item in log))
        self.assertIn(existing, service.items)
        self.assertEqual(verifier.deleted, 1)

    async def test_builder_upload_install_smoke_snapshot_cleanup_and_verify_order(
        self,
    ) -> None:
        log: list[tuple[object, ...]] = []
        _, manifest_bytes, _ = snapshot._capture_runtime_bundle()
        service = FakeSnapshotService(log=log)

        def register_snapshot(name: str) -> None:
            service.items.append(active_snapshot(name, "snapshot-created"))

        builder = FakeSandbox(
            "snapshot-builder",
            log=log,
            on_snapshot=register_snapshot,
        )
        verifier = FakeSandbox(
            "snapshot-verifier",
            log=log,
            manifest=manifest_bytes,
        )

        def factory(params: FakeCreateParams) -> FakeSandbox:
            purpose = params.labels["purpose"]
            if purpose == "snapshot-builder":
                return builder
            if purpose == "snapshot-verification":
                return verifier
            raise AssertionError(f"unexpected purpose: {purpose}")

        client = FakeDaytona(factory, log=log, snapshot_service=service)
        with patched_snapshot(client):
            result = await snapshot.ensure_worker_snapshot("new-snapshot")

        self.assertEqual(result["status"], "created")
        self.assertEqual(result["builder_sandbox_id"], builder.id)
        self.assertEqual(builder.stopped, 1)
        self.assertEqual(builder.created_snapshots, ["new-snapshot"])
        self.assertEqual(builder.deleted, 1)
        self.assertEqual(verifier.deleted, 1)
        self.assertIn(snapshot.RUNTIME_MANIFEST_PATH, builder.files)
        self.assertIn(
            f"{orchestrator.REMOTE_WORK_DIR}/{snapshot.WORKER_REQUIREMENTS_FILE}",
            builder.files,
        )
        self.assertNotIn(
            f"{orchestrator.REMOTE_WORK_DIR}/requirements.txt",
            builder.files,
        )

        def position(predicate: object) -> int:
            for index, item in enumerate(log):
                if predicate(item):
                    return index
            self.fail(f"missing expected operation in log: {log}")

        folder = position(lambda item: item[:2] == ("folder", builder.id))
        install = position(
            lambda item: item[:2] == ("exec", builder.id)
            and str(item[2]).startswith("python3 -m pip")
        )
        self.assertEqual(
            log[install][2],
            "python3 -m pip install --no-cache-dir "
            f"-r {snapshot.WORKER_REQUIREMENTS_FILE}",
        )
        smoke_job = position(
            lambda item: item[:3]
            == ("upload", builder.id, orchestrator.REMOTE_JOB_PATH)
        )
        smoke_exec = position(
            lambda item: item[:3]
            == ("exec", builder.id, orchestrator.WORKER_COMMAND)
        )
        remove_job = position(
            lambda item: item
            == ("delete_file", builder.id, orchestrator.REMOTE_JOB_PATH)
        )
        stop = position(lambda item: item[:2] == ("stop", builder.id))
        make_snapshot = position(
            lambda item: item[:2] == ("create_snapshot", builder.id)
        )
        delete_builder = position(lambda item: item[:2] == ("delete", builder.id))
        create_verifier = position(
            lambda item: item[:2] == ("create", "snapshot-verification")
        )
        ordered_operations = (
            folder,
            install,
            smoke_job,
            smoke_exec,
            remove_job,
            stop,
            make_snapshot,
            delete_builder,
            create_verifier,
        )
        self.assertTrue(
            all(
                earlier < later
                for earlier, later in zip(
                    ordered_operations,
                    ordered_operations[1:],
                )
            ),
            ordered_operations,
        )

    async def test_builder_cleanup_failure_is_reported_after_snapshot_creation(
        self,
    ) -> None:
        log: list[tuple[object, ...]] = []
        service = FakeSnapshotService(log=log)

        def register_snapshot(name: str) -> None:
            service.items.append(active_snapshot(name, "snapshot-created"))

        builder = FakeSandbox(
            "snapshot-builder",
            log=log,
            delete_failure=True,
            on_snapshot=register_snapshot,
        )
        client = FakeDaytona(lambda params: builder, log=log, snapshot_service=service)

        with patched_snapshot(client):
            with self.assertRaisesRegex(
                snapshot.DaytonaSnapshotError,
                "could not be deleted.*snapshot may have been created",
            ):
                await snapshot.ensure_worker_snapshot("cleanup-failure-snapshot")

        self.assertEqual(builder.created_snapshots, ["cleanup-failure-snapshot"])
        self.assertEqual(builder.delete_attempts, 1)
        self.assertEqual(builder.deleted, 0)
        self.assertEqual(len(client.create_calls), 1)

    async def test_dependency_install_failure_deletes_builder(self) -> None:
        log: list[tuple[object, ...]] = []
        builder = FakeSandbox(
            "failed-install-builder",
            log=log,
            install_failure=True,
        )
        client = FakeDaytona(lambda params: builder, log=log)

        with patched_snapshot(client):
            with self.assertRaisesRegex(
                snapshot.DaytonaSnapshotError,
                "failed-install-builder.*worker dependency installation failed",
            ):
                await snapshot.ensure_worker_snapshot("failed-install-snapshot")

        self.assertEqual(builder.delete_attempts, 1)
        self.assertEqual(builder.deleted, 1)
        self.assertEqual(builder.created_snapshots, [])


if __name__ == "__main__":
    unittest.main()
