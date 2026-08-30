"""Tests for the judge-demo integration and state layer.

These tests never create or claim a real Daytona sandbox. One composition test
uses an explicitly synthetic transport while exercising the actual trainer,
worker bridge, rollout worker, GravityEnv, policy, persistence, and visual
loader. The external SDK path is exercised separately by scripts/e2e_smoke.py.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import replace
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from gravity_gauntlet.demo_controller import (
    IntegrationComponents,
    IntegrationContractError,
    LifecycleCollector,
    build_generation_state,
    checkpoint_model_digest,
    coerce_generation_results,
    run_training_demo,
    save_generation_json,
    validate_training_rollout_contract,
    validate_generation_seeds,
    world_state_from_result,
)
from gravity_gauntlet.demo_state import TrainingState


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


POLICY_VERSION = 4
SEEDS = [18_873 + index for index in range(8)]


def write_fixture_checkpoint(
    path: Path,
    *,
    policy_version: int,
    weight_value: float | None = None,
) -> None:
    """Write a tiny trainer-shaped checkpoint for controller contract tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_version": policy_version,
            "obs_dim": 2,
            "model_state_dict": {
                "fixture.weight": torch.tensor(
                    [float(policy_version) if weight_value is None else weight_value]
                )
            },
            "optimizer_state_dict": {},
        },
        path,
    )


def fixture_rollout(index: int) -> dict[str, object]:
    success = index in {1, 4, 7}
    collision = index in {2, 5}
    if success:
        termination = "success"
    elif collision:
        termination = "planet_collision" if index == 2 else "asteroid_collision"
    else:
        termination = "timeout"
    reward = float(index * 10 - 25)
    return {
        "sandbox_id": f"fixture-sandbox-{index}",
        "seed": SEEDS[index - 1],
        "policy_version": POLICY_VERSION,
        "reward": reward,
        "success": success,
        "termination": termination,
        "steps": index + 2,
        "trajectory": [
            {"step": 0, "x": 10.0, "y": float(index)},
            {"step": 1, "x": 20.0, "y": float(index + 1)},
        ],
        "observations": [[float(index), 0.0], [float(index), 1.0]],
        "actions": [index % 9, (index + 1) % 9],
        "rewards": [-0.1, reward + 0.1],
        "policy_mode": "neural_policy",
        "min_clearance": float(index),
        "mean_speed": float(index + 20),
        "max_speed": float(index + 30),
        "fuel_used": float(index) / 10.0,
        "worker_build": "fixture-only",
    }


def fixture_events() -> dict[int, list[dict[str, object]]]:
    events: dict[int, list[dict[str, object]]] = {}
    for index, seed in enumerate(SEEDS, start=1):
        events[index] = [
            {
                "world": index,
                "seed": seed,
                "state": "CREATING",
                "sandbox_id": None,
            },
            {
                "world": index,
                "seed": seed,
                "state": "LIVE",
                "sandbox_id": f"fixture-sandbox-{index}",
            },
            {
                "world": index,
                "seed": seed,
                "state": "RUNNING",
                "sandbox_id": f"fixture-sandbox-{index}",
            },
            {
                "world": index,
                "seed": seed,
                "state": "RESULT_COLLECTED",
                "sandbox_id": f"fixture-sandbox-{index}",
            },
        ]
    return events


class GenerationMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [fixture_rollout(index) for index in range(1, 9)]
        self.state = build_generation_state(
            generation=4,
            policy_version=POLICY_VERSION,
            seeds=SEEDS,
            results=self.results,
            lifecycle_events=fixture_events(),
            next_policy_version=5,
        )

    def test_eight_world_metrics_and_champion_selection(self) -> None:
        rewards = [float(result["reward"]) for result in self.results]

        self.assertEqual(self.state.world_count, 8)
        self.assertAlmostEqual(self.state.average_reward, sum(rewards) / 8)
        self.assertEqual(self.state.best_reward, max(rewards))
        self.assertEqual(self.state.worst_reward, min(rewards))
        self.assertEqual(self.state.best_world, 8)
        self.assertEqual(self.state.best_sandbox_id, "fixture-sandbox-8")
        self.assertEqual(self.state.status, "COMPLETE")
        self.assertEqual(self.state.champion.seed, SEEDS[7])
        self.assertEqual(
            self.state.champion.trajectory,
            self.results[7]["trajectory"],
        )

    def test_success_collision_and_episode_metrics(self) -> None:
        self.assertAlmostEqual(self.state.success_rate, 3 / 8)
        self.assertAlmostEqual(self.state.collision_rate, 2 / 8)
        self.assertAlmostEqual(self.state.average_episode_length, 6.5)
        self.assertAlmostEqual(self.state.average_min_clearance, 4.5)
        self.assertEqual(self.state.max_speed, 38.0)
        self.assertAlmostEqual(self.state.fuel_used, 3.6)

    def test_every_world_shares_policy_but_has_a_different_seed(self) -> None:
        self.assertEqual({world.policy_version for world in self.state.worlds}, {4})
        self.assertEqual(
            len({world.seed for world in self.state.worlds}),
            self.state.world_count,
        )

    def test_lifecycle_and_useful_worker_fields_are_preserved(self) -> None:
        first = self.state.worlds[0]
        self.assertEqual(
            [event["state"] for event in first.lifecycle],
            ["CREATING", "LIVE", "RUNNING", "RESULT_COLLECTED"],
        )
        self.assertEqual(first.status, "SUCCESS")
        self.assertEqual(first.extra["worker_build"], "fixture-only")
        self.assertEqual(first.actions, [1, 2])
        self.assertTrue(first.extra["observations"])

    def test_generated_output_is_strict_json_and_renderer_ready(self) -> None:
        document = self.state.to_dict()
        encoded = json.dumps(document, allow_nan=False)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["generation"], 4)
        self.assertEqual(decoded["policy_version"], 4)
        self.assertEqual(decoded["policy_version_used"], 4)
        self.assertEqual(decoded["next_policy_version"], 5)
        self.assertEqual(decoded["world_count"], 8)
        self.assertEqual(len(decoded["rollouts"]), 8)
        self.assertEqual(
            decoded["rollouts"][0]["trajectory"],
            decoded["worlds"][0]["trajectory"],
        )
        self.assertEqual(decoded["rollouts"][0]["generation"], 4)
        self.assertEqual(decoded["rollouts"][0]["actions"], [1, 2])
        self.assertEqual(decoded["worlds"][0]["steps"], 3)
        self.assertEqual(decoded["rollouts"][0]["steps"], 3)
        self.assertEqual(decoded["champion"]["sandbox_id"], "fixture-sandbox-8")
        self.assertEqual(decoded["champion"]["actions"], [8, 0])
        self.assertEqual(decoded["champion"]["policy_version"], 4)
        self.assertEqual(decoded["champion"]["trajectory"], self.results[7]["trajectory"])


class ContractToleranceTests(unittest.TestCase):
    def test_missing_optional_rollout_fields_are_not_invented(self) -> None:
        world = world_state_from_result(
            {},
            world_index=1,
            expected_seed=99,
            expected_policy_version=3,
        )

        self.assertEqual(world.seed, 99)
        self.assertEqual(world.policy_version, 3)
        self.assertIsNone(world.sandbox_id)
        self.assertIsNone(world.reward)
        self.assertIsNone(world.termination)
        self.assertEqual(world.trajectory, [])
        self.assertEqual(world.status, "RESULT_COLLECTED")

    def test_partial_metric_fields_use_only_real_values(self) -> None:
        state = build_generation_state(
            generation=0,
            policy_version=1,
            seeds=[10, 11],
            results=[
                {"seed": 10, "policy_version": 1, "reward": 7.0},
                {"seed": 11, "policy_version": 1},
            ],
        )

        self.assertEqual(state.average_reward, 7.0)
        self.assertEqual(state.best_reward, 7.0)
        self.assertEqual(state.best_world, 1)
        self.assertEqual(state.status, "INCOMPLETE")
        self.assertEqual(state.extra["worlds_missing_reward"], 1)
        self.assertEqual(state.extra["worlds_missing_sandbox_id"], 2)
        self.assertEqual(state.extra["worlds_missing_trajectory"], 2)

    def test_mismatched_seed_policy_and_world_count_fail(self) -> None:
        with self.assertRaisesRegex(IntegrationContractError, "returned seed"):
            world_state_from_result(
                {"seed": 2},
                world_index=1,
                expected_seed=1,
                expected_policy_version=1,
            )
        with self.assertRaisesRegex(IntegrationContractError, "returned policy"):
            world_state_from_result(
                {"policy_version": 2},
                world_index=1,
                expected_seed=1,
                expected_policy_version=1,
            )
        with self.assertRaisesRegex(IntegrationContractError, "requested seeds"):
            build_generation_state(
                generation=0,
                policy_version=1,
                seeds=[1, 2],
                results=[{"reward": 1.0}],
            )

    def test_seed_validation_requires_exactly_distinct_worlds(self) -> None:
        self.assertEqual(validate_generation_seeds(SEEDS, worlds=8), SEEDS)
        with self.assertRaisesRegex(IntegrationContractError, "different seed"):
            validate_generation_seeds([1, 1], worlds=2)
        with self.assertRaisesRegex(IntegrationContractError, "for 8 worlds"):
            validate_generation_seeds([1, 2], worlds=8)

    def test_lifecycle_and_result_sandbox_ids_must_match(self) -> None:
        with self.assertRaisesRegex(IntegrationContractError, "lifecycle"):
            world_state_from_result(
                {
                    "sandbox_id": "fixture-result-id",
                    "seed": 8,
                    "policy_version": 1,
                },
                world_index=1,
                expected_seed=8,
                expected_policy_version=1,
                lifecycle=[
                    {
                        "world": 1,
                        "seed": 8,
                        "state": "LIVE",
                        "sandbox_id": "fixture-other-id",
                    }
                ],
            )

    def test_duplicate_sandbox_ids_are_rejected(self) -> None:
        first = fixture_rollout(1)
        second = fixture_rollout(2)
        second["sandbox_id"] = first["sandbox_id"]
        with self.assertRaisesRegex(IntegrationContractError, "duplicate sandbox"):
            build_generation_state(
                generation=0,
                policy_version=POLICY_VERSION,
                seeds=SEEDS[:2],
                results=[first, second],
            )

    def test_trainer_boundary_requires_aligned_observations(self) -> None:
        valid = fixture_rollout(1)
        validate_training_rollout_contract([valid], expected_obs_dim=2)
        missing = dict(valid)
        del missing["observations"]
        with self.assertRaisesRegex(IntegrationContractError, "observations"):
            validate_training_rollout_contract([missing], expected_obs_dim=2)
        wrong_dimension = dict(valid)
        wrong_dimension["observations"] = [[0.0], [1.0]]
        with self.assertRaisesRegex(IntegrationContractError, "dimension"):
            validate_training_rollout_contract(
                [wrong_dimension], expected_obs_dim=2
            )

    def test_trainer_boundary_requires_truthful_identity_outcome_and_mode(self) -> None:
        valid = fixture_rollout(1)
        validate_training_rollout_contract(
            [valid],
            expected_obs_dim=2,
            expected_seeds=[SEEDS[0]],
            expected_policy_version=POLICY_VERSION,
        )

        mutations = (
            ("missing seed", lambda value: value.pop("seed"), "seed"),
            (
                "wrong policy",
                lambda value: value.__setitem__("policy_version", 3),
                "expected v4",
            ),
            (
                "string success",
                lambda value: value.__setitem__("success", "false"),
                "JSON boolean",
            ),
            (
                "wrong mode",
                lambda value: value.__setitem__("policy_mode", "seeded_random_v0"),
                "policy_mode",
            ),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label):
                changed = dict(valid)
                mutate(changed)
                with self.assertRaisesRegex(IntegrationContractError, message):
                    validate_training_rollout_contract(
                        [changed],
                        expected_obs_dim=2,
                        expected_seeds=[SEEDS[0]],
                        expected_policy_version=POLICY_VERSION,
                    )

    def test_result_envelopes_are_supported_but_fake_objects_are_rejected(self) -> None:
        values = [{"reward": 1.0}]
        self.assertEqual(coerce_generation_results({"results": values}), values)
        self.assertEqual(coerce_generation_results({"rollouts": values}), values)
        with self.assertRaises(IntegrationContractError):
            coerce_generation_results({"not_results": values})


class LifecycleAndHistoryTests(unittest.TestCase):
    def test_callback_collects_lifecycle_by_one_based_world(self) -> None:
        collector = LifecycleCollector(echo=False)
        collector(
            {
                "world": 2,
                "seed": 100,
                "state": "LIVE",
                "sandbox_id": "real-abc",
            }
        )
        collector(
            {
                "world": 2,
                "seed": 100,
                "state": "RESULT_COLLECTED",
                "sandbox_id": "real-abc",
                "reward": 5.0,
            }
        )

        self.assertEqual(
            [event["state"] for event in collector.events_for(2)],
            ["LIVE", "RESULT_COLLECTED"],
        )
        self.assertEqual(collector.events_for(2)[0]["sandbox_id"], "real-abc")

    def test_training_history_exposes_ghosts_and_respects_limit(self) -> None:
        training = TrainingState(history_limit=2)
        for generation in range(3):
            result = fixture_rollout(1)
            result["seed"] = generation + 50
            result["policy_version"] = generation + 1
            state = build_generation_state(
                generation=generation,
                policy_version=generation + 1,
                seeds=[generation + 50],
                results=[result],
                next_policy_version=generation + 2,
            )
            training.add_generation(state)

        self.assertEqual(training.current_generation, 2)
        self.assertEqual(training.current_policy_version, 4)
        self.assertEqual(training.total_worlds_run, 3)
        self.assertEqual(
            [item.generation for item in training.recent_generations], [1, 2]
        )
        ghosts = training.to_dict()["ghost_history"]
        self.assertEqual([item["generation"] for item in ghosts], [1, 2])
        self.assertTrue(ghosts[-1]["champion"]["trajectory"])
        visual_rollouts = training.to_dict()["rollouts"]
        self.assertEqual(len(visual_rollouts), 2)
        self.assertEqual(
            [rollout["generation"] for rollout in visual_rollouts], [1, 2]
        )

    def test_generation_file_has_exact_name_and_expected_topology(self) -> None:
        result = fixture_rollout(1)
        result["seed"] = 77
        result["policy_version"] = 2
        state = build_generation_state(
            generation=12,
            policy_version=2,
            seeds=[77],
            results=[result],
            next_policy_version=3,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_generation_json(state, directory)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(path.name, "generation_012.json")
        self.assertEqual(payload["world_count"], 1)
        self.assertEqual(payload["best_world"], 1)
        self.assertEqual(payload["best_sandbox_id"], "fixture-sandbox-1")
        self.assertEqual(len(payload["worlds"]), 1)
        self.assertEqual(
            payload["rollouts"][0]["trajectory"],
            payload["worlds"][0]["trajectory"],
        )
        self.assertIn("champion", payload)

    def test_saved_generation_is_directly_consumable_by_visual_loader(self) -> None:
        from visual_demo import load_rollout_trails

        result = fixture_rollout(1)
        result["seed"] = 77
        result["policy_version"] = 2
        state = build_generation_state(
            generation=0,
            policy_version=2,
            seeds=[77],
            results=[result],
            next_policy_version=3,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_generation_json(state, directory)
            attempts = load_rollout_trails(path)

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].sandbox_id, "fixture-sandbox-1")
        self.assertEqual(attempts[0].policy_version, 2)
        self.assertEqual(attempts[0].points, ((10.0, 1.0), (20.0, 2.0)))

    def test_fixture_replay_validates_geometry_without_claiming_daytona(self) -> None:
        from gravity_env import GravityEnv
        from visual_demo import _replay_environment, load_rollout_trails

        seed = 77
        sandbox_id = "fixture-sandbox-release-proof-77"
        result = fixture_rollout(1)
        result.update(
            {
                "sandbox_id": sandbox_id,
                "seed": seed,
                "policy_version": 0,
                "policy_mode": "seeded_random_v0",
                "action_vectors": [[0.0, 0.0], [1.0, 0.0]],
                "universe": GravityEnv(seed=seed).universe_dict(),
            }
        )
        lifecycle = {
            1: [
                {"world": 1, "seed": seed, "state": "CREATING", "sandbox_id": None},
                {"world": 1, "seed": seed, "state": "LIVE", "sandbox_id": sandbox_id},
                {"world": 1, "seed": seed, "state": "RUNNING", "sandbox_id": sandbox_id},
                {"world": 1, "seed": seed, "state": "SUCCESS", "sandbox_id": sandbox_id},
                {
                    "world": 1,
                    "seed": seed,
                    "state": "RESULT_COLLECTED",
                    "sandbox_id": sandbox_id,
                },
            ]
        }
        state = build_generation_state(
            generation=0,
            policy_version=0,
            seeds=[seed],
            results=[result],
            lifecycle_events=lifecycle,
            next_policy_version=1,
            execution_backend="fixture",
            extra={
                "execution_backend": "fixture",
                "seed_batch": [seed],
                "trainer_checkpoint": "checkpoints/policy_v001.pt",
                "training": {"loss": 0.25},
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            attempts = load_rollout_trails(save_generation_json(state, directory))

        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0].daytona_verified)
        self.assertEqual(attempts[0].provenance, "UNVERIFIED RECORDED REPLAY")
        self.assertEqual(
            _replay_environment(attempts[0]).universe_dict(),
            result["universe"],
        )

        mismatched = dict(attempts[0].universe or {})
        mismatched["seed"] = seed + 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            _replay_environment(replace(attempts[0], universe=mismatched))


class FixtureBoundaryTests(unittest.TestCase):
    """Controller-loop unit checks; these are not real Daytona proof."""

    def _components(
        self,
        events: list[str],
        *,
        fail_daytona: bool = False,
        incomplete_daytona: bool = False,
        unchanged_weights: bool = False,
        record_generation_delta: int = 0,
        record_seed_delta: int = 0,
    ) -> IntegrationComponents:
        async def run_daytona_training(**kwargs: object) -> list[dict[str, object]]:
            checkpoint_dir = Path(kwargs["checkpoint_dir"])
            write_fixture_checkpoint(
                checkpoint_dir / "policy_v000.pt",
                policy_version=0,
            )
            events.append("daytona_called")
            if fail_daytona:
                raise RuntimeError("fixture Daytona failure")
            history: list[dict[str, object]] = []
            for generation in range(int(kwargs["generations"])):
                policy_version = generation
                seeds = [
                    int(kwargs["base_seed"]) + generation * 100 + index
                    for index in range(int(kwargs["worlds"]))
                ]
                results: list[dict[str, object]] = []
                for index, seed in enumerate(seeds, start=1):
                    sandbox_id = f"fixture-boundary-{generation}-{index}"
                    for state in ("CREATING", "LIVE", "RUNNING", "RESULT_COLLECTED"):
                        kwargs["event_callback"](
                            {
                                "generation": generation,
                                "policy_version": policy_version,
                                "world": index,
                                "seed": seed,
                                "state": state,
                                "sandbox_id": (
                                    None if state == "CREATING" else sandbox_id
                                ),
                            }
                        )
                    rollout = fixture_rollout(index)
                    rollout.update(
                        {
                            "sandbox_id": sandbox_id,
                            "seed": seed,
                            "policy_version": policy_version,
                            "policy_mode": (
                                "seeded_random_v0"
                                if policy_version == 0
                                else "neural_policy"
                            ),
                        }
                    )
                    if incomplete_daytona and index == 1:
                        rollout["trajectory"] = rollout["trajectory"][:1]
                    results.append(rollout)
                events.append("all_results_collected")
                validation = kwargs["rollout_validator"](
                    policy_version,
                    seeds,
                    results,
                )
                if asyncio.iscoroutine(validation):
                    await validation
                events.append("trainer_updated")
                next_version = policy_version + 1
                checkpoint = checkpoint_dir / f"policy_v{next_version:03d}.pt"
                write_fixture_checkpoint(
                    checkpoint,
                    policy_version=next_version,
                    weight_value=(
                        float(policy_version) if unchanged_weights else None
                    ),
                )
                record: dict[str, object] = {
                    "generation": generation + record_generation_delta,
                    "policy_version": policy_version,
                    "next_policy_version": next_version,
                    "seeds": [seed + record_seed_delta for seed in seeds],
                    "checkpoint": str(checkpoint),
                    "worlds": len(results),
                    "episodes": len(results),
                    "loss": 0.25,
                }
                persisted = kwargs["on_generation"](record, results)
                if asyncio.iscoroutine(persisted):
                    await persisted
                history.append(record)
            return history

        return IntegrationComponents(
            run_daytona_training=run_daytona_training,
        )

    def test_fixture_loop_calls_one_generation_then_trains_and_saves(self) -> None:
        events: list[str] = []
        components = self._components(events)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                training = asyncio.run(
                    run_training_demo(
                        generations=1,
                        worlds=2,
                        max_steps=5,
                        base_seed=900,
                        obs_dim=2,
                        runs_dir=root / "runs",
                        checkpoint_dir=root / "checkpoints",
                        components=components,
                        echo_lifecycle=False,
                    )
                )
            generation_path = root / "runs" / "generation_000.json"
            initial_checkpoint = root / "checkpoints" / "policy_v000.pt"
            next_checkpoint = root / "checkpoints" / "policy_v001.pt"

            self.assertTrue(generation_path.is_file())
            self.assertTrue(initial_checkpoint.is_file())
            self.assertTrue(next_checkpoint.is_file())
            input_digest = checkpoint_model_digest(
                initial_checkpoint,
                expected_policy_version=0,
            )
            next_digest = checkpoint_model_digest(
                next_checkpoint,
                expected_policy_version=1,
            )
            payload = json.loads(generation_path.read_text(encoding="utf-8"))

        self.assertEqual(events.count("daytona_called"), 1)
        self.assertLess(events.index("all_results_collected"), events.index("trainer_updated"))
        self.assertEqual(training.current_policy_version, 1)
        self.assertEqual(payload["policy_version"], 0)
        self.assertEqual(payload["next_policy_version"], 1)
        self.assertEqual(payload["extra"]["training"]["episodes"], 2)
        self.assertEqual(payload["extra"]["execution_backend"], "fixture")
        self.assertEqual(
            {world["execution_backend"] for world in payload["worlds"]},
            {"fixture"},
        )
        self.assertNotEqual(input_digest, next_digest)
        self.assertEqual(
            payload["extra"]["policy_update"],
            {
                "input_checkpoint": str(initial_checkpoint),
                "next_checkpoint": str(next_checkpoint),
                "input_model_sha256": input_digest,
                "next_model_sha256": next_digest,
                "weights_changed": True,
            },
        )
        self.assertTrue(all(world["actions"] for world in payload["worlds"]))

    def test_unchanged_policy_weights_are_not_renamed_or_persisted(self) -> None:
        events: list[str] = []
        components = self._components(events, unchanged_weights=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    IntegrationContractError,
                    "unchanged model weights",
                ):
                    asyncio.run(
                        run_training_demo(
                            generations=1,
                            worlds=2,
                            max_steps=5,
                            obs_dim=2,
                            runs_dir=root / "runs",
                            checkpoint_dir=root / "checkpoints",
                            components=components,
                            echo_lifecycle=False,
                        )
                    )
            self.assertFalse((root / "runs" / "generation_000.json").exists())

    def test_trainer_record_generation_and_seed_identity_are_enforced(self) -> None:
        cases = (
            ({"record_generation_delta": 1}, "generation identity is out of order"),
            ({"record_seed_delta": 1}, "seeds do not match"),
        )
        for component_options, message in cases:
            with self.subTest(component_options=component_options):
                events: list[str] = []
                components = self._components(events, **component_options)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(IntegrationContractError, message):
                            asyncio.run(
                                run_training_demo(
                                    generations=1,
                                    worlds=2,
                                    max_steps=5,
                                    obs_dim=2,
                                    runs_dir=root / "runs",
                                    checkpoint_dir=root / "checkpoints",
                                    components=components,
                                    echo_lifecycle=False,
                                )
                            )
                    self.assertFalse(
                        (root / "runs" / "generation_000.json").exists()
                    )

    def test_more_generations_than_history_limit_complete_successfully(self) -> None:
        events: list[str] = []
        components = self._components(events)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                training = asyncio.run(
                    run_training_demo(
                        generations=13,
                        worlds=1,
                        max_steps=5,
                        obs_dim=2,
                        runs_dir=root / "runs",
                        checkpoint_dir=root / "checkpoints",
                        components=components,
                        echo_lifecycle=False,
                    )
                )

            self.assertEqual(training.current_generation, 12)
            self.assertEqual(training.current_policy_version, 13)
            self.assertEqual(training.total_worlds_run, 13)
            self.assertEqual(len(training.recent_generations), 12)
            self.assertTrue((root / "runs" / "generation_012.json").is_file())

    def test_fixture_daytona_failure_never_trains_or_saves_generation(self) -> None:
        events: list[str] = []
        components = self._components(events, fail_daytona=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    IntegrationContractError, "no local rollout fallback"
                ):
                    asyncio.run(
                        run_training_demo(
                            generations=1,
                            worlds=2,
                            max_steps=5,
                            obs_dim=2,
                            runs_dir=root / "runs",
                            checkpoint_dir=root / "checkpoints",
                            components=components,
                            echo_lifecycle=False,
                        )
                    )
            self.assertFalse((root / "runs" / "generation_000.json").exists())

        self.assertEqual(events.count("daytona_called"), 1)
        self.assertNotIn("trainer_updated", events)

    def test_incomplete_daytona_result_never_updates_policy(self) -> None:
        events: list[str] = []
        components = self._components(events, incomplete_daytona=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    IntegrationContractError, "at least two points"
                ):
                    asyncio.run(
                        run_training_demo(
                            generations=1,
                            worlds=2,
                            max_steps=5,
                            obs_dim=2,
                            runs_dir=root / "runs",
                            checkpoint_dir=root / "checkpoints",
                            components=components,
                            echo_lifecycle=False,
                        )
                    )
            self.assertFalse((root / "runs" / "generation_000.json").exists())
            self.assertFalse((root / "checkpoints" / "policy_v001.pt").exists())

        self.assertEqual(events.count("daytona_called"), 1)
        self.assertNotIn("trainer_updated", events)

    def test_existing_generated_target_blocks_before_daytona_call(self) -> None:
        events: list[str] = []
        components = self._components(events)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs_dir = root / "runs"
            runs_dir.mkdir()
            (runs_dir / "generation_000.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(IntegrationContractError, "overwrite"):
                asyncio.run(
                    run_training_demo(
                        generations=1,
                        worlds=2,
                        max_steps=5,
                        obs_dim=2,
                        runs_dir=runs_dir,
                        checkpoint_dir=root / "checkpoints",
                        components=components,
                        echo_lifecycle=False,
                    )
                )

        self.assertNotIn("daytona_called", events)


class ActualComponentPipelineTests(unittest.TestCase):
    """Compose the real trainer/worker/policy/state/visual path offline.

    Only the Daytona SDK transport is substituted. The fake transport invokes
    the actual Daytona worker bridge, so this catches schema drift between the
    real neighbouring components without claiming a real sandbox execution.
    """

    def test_two_generation_v0_to_v1_pipeline_persists_visual_rollouts(self) -> None:
        try:
            import trainer
            from daytona_worker_entry import execute_daytona_job
            from visual_demo import load_rollout_trails
        except ImportError as exc:
            self.skipTest(f"full Python integration dependencies unavailable: {exc}")

        jobs: list[dict[str, object]] = []

        async def fake_daytona_transport(**kwargs: object) -> list[dict[str, object]]:
            policy_version = int(kwargs["policy_version"])
            callback = kwargs.get("event_callback")
            results: list[dict[str, object]] = []

            async def emit(event: dict[str, object]) -> None:
                if callback is None:
                    return
                emitted = callback(event)
                if asyncio.iscoroutine(emitted):
                    await emitted

            for index, seed in enumerate(kwargs["seeds"], start=1):
                sandbox_id = f"fixture-offline-{policy_version}-{index}"
                job = {
                    "sandbox_id": sandbox_id,
                    "seed": int(seed),
                    "policy_version": policy_version,
                    "policy_weights": kwargs["policy_weights"],
                    "max_steps": int(kwargs["max_steps"]),
                }
                jobs.append(dict(job))
                await emit(
                    {
                        "world": index,
                        "seed": int(seed),
                        "state": "CREATING",
                        "sandbox_id": None,
                    }
                )
                for state in ("LIVE", "RUNNING"):
                    await emit(
                        {
                            "world": index,
                            "seed": int(seed),
                            "state": state,
                            "sandbox_id": sandbox_id,
                        }
                    )
                result = execute_daytona_job(job)
                terminal_state = (
                    "SUCCESS"
                    if result["success"]
                    else "COLLISION"
                    if "collision" in str(result["termination"])
                    else str(result["termination"]).upper()
                )
                await emit(
                    {
                        "world": index,
                        "seed": int(seed),
                        "state": terminal_state,
                        "sandbox_id": sandbox_id,
                    }
                )
                await emit(
                    {
                        "world": index,
                        "seed": int(seed),
                        "state": "RESULT_COLLECTED",
                        "sandbox_id": sandbox_id,
                        "reward": result["reward"],
                    }
                )
                results.append(result)
            return results

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    trainer,
                    "_load_daytona_run_generation",
                    return_value=fake_daytona_transport,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                training = asyncio.run(
                    run_training_demo(
                        generations=2,
                        worlds=2,
                        max_steps=2,
                        base_seed=321,
                        runs_dir=root / "runs",
                        checkpoint_dir=root / "checkpoints",
                        components=IntegrationComponents(
                            run_daytona_training=trainer.run_daytona_training
                        ),
                        echo_lifecycle=False,
                    )
                )

            first_path = root / "runs" / "generation_000.json"
            second_path = root / "runs" / "generation_001.json"
            first = json.loads(first_path.read_text(encoding="utf-8"))
            second = json.loads(second_path.read_text(encoding="utf-8"))
            attempts = load_rollout_trails(second_path)

            self.assertEqual(training.current_policy_version, 2)
            self.assertEqual([job["policy_version"] for job in jobs], [0, 0, 1, 1])
            self.assertTrue(all(job["policy_weights"] is None for job in jobs[:2]))
            self.assertTrue(
                all(
                    isinstance(job["policy_weights"], str)
                    and bool(job["policy_weights"])
                    for job in jobs[2:]
                )
            )
            self.assertEqual(
                {world["extra"]["policy_mode"] for world in first["worlds"]},
                {"seeded_random_v0"},
            )
            self.assertEqual(
                {world["extra"]["policy_mode"] for world in second["worlds"]},
                {"neural_policy"},
            )
            self.assertTrue(all(world["actions"] for world in second["worlds"]))
            self.assertTrue(all(rollout["actions"] for rollout in second["rollouts"]))
            self.assertEqual(len(attempts), 2)
            self.assertEqual({attempt.policy_version for attempt in attempts}, {1})
            self.assertTrue(all(not attempt.daytona_verified for attempt in attempts))
            self.assertTrue((root / "checkpoints" / "policy_v000.pt").is_file())
            self.assertTrue((root / "checkpoints" / "policy_v001.pt").is_file())
            self.assertTrue((root / "checkpoints" / "policy_v002.pt").is_file())


if __name__ == "__main__":
    unittest.main()
