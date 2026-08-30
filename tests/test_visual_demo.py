"""UI-owned contract tests for truthful generation replay."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gravity_env import GravityEnv
from visual_demo import (
    AttemptTrail,
    _ghost_attempts,
    _initial_replay_index,
    _learning_history,
    _lifecycle_states,
    _replay_environment,
    _select_replay_group,
    _validated_replay_environments,
    load_rollout_trails,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class VisualReplayContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_root = Path(self.temp_dir.name)

    def _write(self, payload: object, name: str = "generation.json") -> Path:
        path = self.temp_root / name
        path.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _world(
        *,
        generation: int,
        policy_version: int,
        world_index: int,
        seed: int,
        reward: float,
        success: bool,
    ) -> dict[str, object]:
        env = GravityEnv(seed=seed)
        x, y = map(float, env.ship_position)
        termination = "success" if success else "planet_collision"
        terminal_state = "SUCCESS" if success else "COLLISION"
        sandbox_id = f"sandbox-{generation}-{world_index}-7f61525c9e84"
        trajectory = [
            {
                "step": 0,
                "x": x,
                "y": y,
                "vx": 0.0,
                "vy": 0.0,
                "reward": 0.0,
                "clearance": 120.0,
            },
            {
                "step": 4,
                "x": x + 8.0,
                "y": y - 4.0,
                "vx": 2.0,
                "vy": -1.0,
                "reward": reward,
                "clearance": 90.0,
            },
        ]
        lifecycle = []
        for state in ("CREATING", "LIVE", "RUNNING", terminal_state, "RESULT_COLLECTED"):
            event: dict[str, object] = {
                "generation": generation,
                "policy_version": policy_version,
                "world": world_index,
                "seed": seed,
                "sandbox_id": None if state == "CREATING" else sandbox_id,
                "state": state,
            }
            if state in {terminal_state, "RESULT_COLLECTED"}:
                event["reward"] = reward
            if state == terminal_state:
                event["termination"] = termination
            lifecycle.append(event)
        return {
            "world_index": world_index,
            "seed": seed,
            "policy_version": policy_version,
            "sandbox_id": sandbox_id,
            "status": terminal_state,
            "reward": reward,
            "success": success,
            "termination": termination,
            "trajectory": trajectory,
            "actions": [0],
            "execution_backend": "daytona",
            "min_clearance": 90.0,
            "episode_length": 1,
            "mean_speed": math.sqrt(5.0),
            "max_speed": math.sqrt(5.0),
            "fuel_used": 0.0,
            "lifecycle": lifecycle,
            "extra": {
                "universe": env.universe_dict(),
                "action_vectors": [[0.0, 0.0]],
            },
        }

    @classmethod
    def _generation(
        cls,
        *,
        generation: int = 0,
        rewards: tuple[float, float] = (-25.0, 80.0),
        successes: tuple[bool, bool] = (False, True),
    ) -> dict[str, object]:
        policy_version = generation
        seeds = (101 + generation * 10, 102 + generation * 10)
        worlds = [
            cls._world(
                generation=generation,
                policy_version=policy_version,
                world_index=index + 1,
                seed=seeds[index],
                reward=rewards[index],
                success=successes[index],
            )
            for index in range(2)
        ]
        champion = max(worlds, key=lambda world: float(world["reward"]))
        average_reward = math.fsum(rewards) / len(rewards)
        success_rate = sum(successes) / len(successes)
        collision_rate = sum(not success for success in successes) / len(successes)
        return {
            "generation": generation,
            "policy_version": policy_version,
            "policy_version_used": policy_version,
            "next_policy_version": policy_version + 1,
            "status": "COMPLETE",
            "world_count": len(worlds),
            "worlds": worlds,
            "average_reward": average_reward,
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "average_episode_length": 1.0,
            "best_world": champion["world_index"],
            "best_sandbox_id": champion["sandbox_id"],
            "champion": {
                "world_index": champion["world_index"],
                "seed": champion["seed"],
                "sandbox_id": champion["sandbox_id"],
                "policy_version": policy_version,
                "reward": champion["reward"],
                "success": champion["success"],
                "termination": champion["termination"],
                "trajectory": champion["trajectory"],
                "actions": champion["actions"],
                "execution_backend": "daytona",
                "generation": generation,
            },
            "extra": {
                "execution_backend": "daytona",
                "seed_batch": list(seeds),
                "trainer_checkpoint": f"checkpoints/policy_v{policy_version + 1:03d}.pt",
                "training": {"loss": 0.125, "episodes": 2},
                "policy_update": {
                    "input_checkpoint": f"checkpoints/policy_v{policy_version:03d}.pt",
                    "next_checkpoint": f"checkpoints/policy_v{policy_version + 1:03d}.pt",
                    "input_model_sha256": "a" * 64,
                    "next_model_sha256": "b" * 64,
                    "weights_changed": True,
                },
            },
        }

    @classmethod
    def _raw_generation(cls) -> dict[str, object]:
        controller = cls._generation()
        results = []
        for world in controller["worlds"]:
            results.append(
                {
                    "sandbox_id": world["sandbox_id"],
                    "seed": world["seed"],
                    "policy_version": world["policy_version"],
                    "reward": world["reward"],
                    "success": world["success"],
                    "termination": world["termination"],
                    "trajectory": world["trajectory"],
                    "actions": world["actions"],
                    "action_vectors": world["extra"]["action_vectors"],
                    "universe": world["extra"]["universe"],
                }
            )
        rewards = [float(result["reward"]) for result in results]
        champion = max(results, key=lambda result: float(result["reward"]))
        return {
            "summary": {
                "worlds": len(results),
                "successful": sum(bool(result["success"]) for result in results),
                "average_reward": math.fsum(rewards) / len(rewards),
                "best_reward": max(rewards),
                "best_sandbox": champion["sandbox_id"],
                "total_trajectory_points": sum(
                    len(result["trajectory"]) for result in results
                ),
                "concurrent": True,
                "wall_clock_seconds": 1.25,
                "seeds": [result["seed"] for result in results],
                "sandbox_ids": [result["sandbox_id"] for result in results],
                "cleanup": "explicit_delete_confirmed",
            },
            "results": results,
        }

    def test_loads_complete_controller_generation_contract(self) -> None:
        payload = self._generation()
        attempts = load_rollout_trails(self._write(payload))

        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(attempt.daytona_verified for attempt in attempts))
        self.assertEqual({attempt.provenance for attempt in attempts}, {"DAYTONA TRAINING"})
        self.assertEqual([attempt.sandbox_id for attempt in attempts], [
            payload["worlds"][0]["sandbox_id"],
            payload["worlds"][1]["sandbox_id"],
        ])
        self.assertEqual(attempts[0].generation, 0)
        self.assertEqual(attempts[0].policy_version, 0)
        self.assertEqual(attempts[0].next_policy_version, 1)
        self.assertEqual(attempts[0].generation_average_reward, 27.5)
        self.assertEqual(attempts[0].generation_best_reward, 80.0)
        self.assertEqual(attempts[0].generation_success_rate, 0.5)
        self.assertEqual(attempts[0].generation_collision_rate, 0.5)
        self.assertEqual(attempts[0].generation_world_count, 2)
        self.assertEqual(
            _lifecycle_states(attempts[0]),
            ("CREATING", "LIVE", "RUNNING", "COLLISION", "RESULT_COLLECTED"),
        )
        self.assertEqual(attempts[0].action_vectors, ((0.0, 0.0),))
        self.assertEqual(_replay_environment(attempts[0]).universe_dict(), attempts[0].universe)

    def test_champion_is_unique_maximum_and_default_replay(self) -> None:
        attempts = load_rollout_trails(self._write(self._generation()))
        champions = [attempt for attempt in attempts if attempt.is_champion]

        self.assertEqual(len(champions), 1)
        self.assertEqual(champions[0].reward, 80.0)
        self.assertEqual(champions[0].world_index, 2)
        self.assertEqual(_initial_replay_index(attempts, None), attempts.index(champions[0]))

        conflicting = self._generation()
        conflicting["champion"] = dict(conflicting["champion"])
        conflicting["champion"]["sandbox_id"] = conflicting["worlds"][0]["sandbox_id"]
        conflicting["champion"]["seed"] = conflicting["worlds"][0]["seed"]
        conflicting["champion"]["world_index"] = conflicting["worlds"][0]["world_index"]
        conflicting["champion"]["reward"] = conflicting["worlds"][0]["reward"]
        conflicting["champion"]["success"] = conflicting["worlds"][0]["success"]
        conflicting["champion"]["termination"] = conflicting["worlds"][0]["termination"]
        conflicting["champion"]["trajectory"] = conflicting["worlds"][0]["trajectory"]
        conflicting["champion"]["actions"] = conflicting["worlds"][0]["actions"]
        conflicting["best_sandbox_id"] = conflicting["worlds"][0]["sandbox_id"]
        conflicting["best_world"] = conflicting["worlds"][0]["world_index"]
        conflicting["best_reward"] = conflicting["worlds"][0]["reward"]
        with self.assertRaisesRegex(ValueError, "champion is not a maximum recorded reward"):
            load_rollout_trails(self._write(conflicting, "conflicting.json"))

    def test_lifecycle_order_and_identity_are_provenance_gates(self) -> None:
        mutations = {
            "missing creating": lambda events: events.pop(0),
            "reordered": lambda events: events.__setitem__(slice(0, 2), [events[1], events[0]]),
            "wrong id": lambda events: events[1].__setitem__("sandbox_id", "another-sandbox"),
            "missing live id": lambda events: events[1].pop("sandbox_id"),
            "wrong seed": lambda events: events[1].__setitem__("seed", 999),
            "missing seed": lambda events: events[1].pop("seed"),
            "wrong generation": lambda events: events[1].__setitem__("generation", 99),
            "missing generation": lambda events: events[1].pop("generation"),
            "wrong policy": lambda events: events[1].__setitem__("policy_version", 99),
            "missing policy": lambda events: events[1].pop("policy_version"),
            "wrong world": lambda events: events[1].__setitem__("world", 99),
            "missing world": lambda events: events[1].pop("world"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = deepcopy(self._generation())
                mutate(payload["worlds"][0]["lifecycle"])
                attempts = load_rollout_trails(self._write(payload, f"{label}.json"))
                self.assertFalse(any(attempt.daytona_verified for attempt in attempts))
                self.assertEqual(
                    attempts[0].sandbox_id,
                    payload["worlds"][0]["sandbox_id"],
                )

    def test_empty_and_invalid_trajectories_fail_closed(self) -> None:
        for index, payload in enumerate((
            {"worlds": []},
            {"rollouts": []},
            {"recent_generations": []},
        )):
            with self.subTest(empty=index):
                with self.assertRaisesRegex(ValueError, "no recorded trajectories"):
                    load_rollout_trails(self._write(payload, f"empty-{index}.json"))

        invalid = (
            {"trajectory": []},
            {"trajectory": [{"x": 1.0, "y": 2.0}]},
            {"trajectory": [{"x": 1.0}, {"x": 2.0, "y": 3.0}]},
            {"trajectory": [{"x": 1.0, "y": 2.0}, {"x": math.inf, "y": 3.0}]},
            {
                "trajectory": [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 3.0}],
                "action_vectors": [[2.0, 0.0]],
            },
        )
        for index, payload in enumerate(invalid):
            with self.subTest(invalid=index):
                path = self.temp_root / f"invalid-{index}.json"
                if index == 3:
                    path.write_text('{"trajectory":[{"x":1,"y":2},{"x":1e999,"y":3}]}')
                else:
                    path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_rollout_trails(path)

        contradictory = {
            "trajectory": [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 3.0}],
            "success": True,
            "termination": "planet_collision",
        }
        with self.assertRaisesRegex(ValueError, "different outcomes"):
            load_rollout_trails(self._write(contradictory, "contradictory.json"))

    def test_history_is_retained_but_current_generation_is_partitioned(self) -> None:
        state = {
            "recent_generations": [
                self._generation(generation=0, rewards=(-40.0, 10.0), successes=(False, False)),
                self._generation(generation=1, rewards=(-12.0, 40.0), successes=(False, True)),
            ]
        }
        attempts = load_rollout_trails(self._write(state, "training_state.json"))
        current = _select_replay_group(attempts, generation=None, policy_version=None)
        history = _learning_history(attempts)

        self.assertEqual(len(attempts), 4)
        self.assertEqual({attempt.generation for attempt in current}, {1})
        self.assertEqual([item["generation"] for item in history], [0, 1])
        self.assertEqual([item["average_reward"] for item in history], [-15.0, 14.0])
        self.assertEqual([item["success_rate"] for item in history], [0.0, 0.5])

    def test_ghost_selection_is_seed_safe_and_never_reveals_active_future(self) -> None:
        old_same_seed = AttemptTrail(
            points=((10.0, 10.0), (20.0, 20.0)),
            reward=-5.0,
            success=False,
            seed=7,
            generation=0,
        )
        old_other_seed = AttemptTrail(
            points=((10.0, 10.0), (20.0, 20.0)),
            reward=1.0,
            success=True,
            seed=8,
            generation=0,
        )
        active = AttemptTrail(
            points=((11.0, 11.0), (30.0, 30.0)),
            reward=9.0,
            success=True,
            seed=7,
            generation=1,
        )

        ghosts = _ghost_attempts([old_same_seed, old_other_seed, active], 7, active)
        self.assertEqual(ghosts, [old_same_seed])
        self.assertNotIn(active, ghosts)
        self.assertNotIn(old_other_seed, ghosts)

        unknown_generation = replace(active, generation=None)
        unknown_peer = replace(old_same_seed, generation=None)
        self.assertEqual(
            _ghost_attempts([unknown_peer, unknown_generation], 7, unknown_generation),
            [],
        )

    def test_recorded_replay_requires_recorded_universe_geometry(self) -> None:
        attempt = AttemptTrail(
            points=((10.0, 10.0), (20.0, 20.0)),
            reward=1.0,
            success=False,
            seed=7,
            sandbox_id="sandbox-without-universe",
            termination="timeout",
            provenance="UNVERIFIED RECORDED REPLAY",
        )

        with self.assertRaisesRegex(ValueError, "no recorded universe"):
            _replay_environment(attempt)

    def test_stale_historical_ghost_universe_fails_before_render(self) -> None:
        current_env = GravityEnv(seed=7)
        active = AttemptTrail(
            points=((10.0, 10.0), (20.0, 20.0)),
            reward=2.0,
            success=False,
            seed=7,
            generation=1,
            universe=current_env.universe_dict(),
            termination="timeout",
            provenance="UNVERIFIED RECORDED REPLAY",
        )
        stale_universe = deepcopy(current_env.universe_dict())
        stale_universe["portal"]["position"][0] += 1.0
        historical = replace(active, generation=0, universe=stale_universe)

        with self.assertRaisesRegex(ValueError, "does not match"):
            _validated_replay_environments([historical, active])

    def test_policy_update_proof_is_required_for_daytona_training_label(self) -> None:
        payload = self._generation()
        del payload["extra"]["policy_update"]
        attempts = load_rollout_trails(self._write(payload, "no-update-proof.json"))

        self.assertFalse(any(attempt.daytona_verified for attempt in attempts))
        self.assertEqual(
            {attempt.provenance for attempt in attempts},
            {"UNVERIFIED RECORDED REPLAY"},
        )

    def test_controller_identity_and_checkpoint_fields_are_provenance_gates(self) -> None:
        def generation_gap(payload: dict[str, object]) -> None:
            payload["generation"] = 7
            payload["champion"]["generation"] = 7
            for world in payload["worlds"]:
                for event in world["lifecycle"]:
                    event["generation"] = 7

        mutations = {
            "missing policy alias": lambda payload: payload.pop("policy_version_used"),
            "wrong policy alias": lambda payload: payload.__setitem__(
                "policy_version_used", 9
            ),
            "generation policy gap": generation_gap,
            "unbound trainer checkpoint": lambda payload: payload["extra"].__setitem__(
                "trainer_checkpoint", "other/policy_v001.pt"
            ),
            "wrong input checkpoint": lambda payload: payload["extra"][
                "policy_update"
            ].__setitem__("input_checkpoint", "checkpoints/not-policy-v0.pt"),
            "noncanonical world index": lambda payload: payload["worlds"][0].__setitem__(
                "world_index", 9
            ),
            "invalid action index": lambda payload: payload["worlds"][0].__setitem__(
                "actions", [999]
            ),
            "nonterminal world status": lambda payload: payload["worlds"][0].__setitem__(
                "status", "ERROR"
            ),
            "conflicting champion outcome": lambda payload: payload[
                "champion"
            ].__setitem__("success", False),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = deepcopy(self._generation())
                mutate(payload)
                attempts = load_rollout_trails(
                    self._write(payload, f"controller-{label.replace(' ', '-')}.json")
                )
                self.assertFalse(any(attempt.daytona_verified for attempt in attempts))

    def test_raw_daytona_summary_identity_is_a_provenance_gate(self) -> None:
        baseline = load_rollout_trails(self._write(self._raw_generation(), "raw.json"))
        self.assertTrue(all(attempt.daytona_verified for attempt in baseline))
        self.assertEqual({attempt.provenance for attempt in baseline}, {"DAYTONA ROLLOUT"})

        mutations = {
            "wrong seeds": lambda payload: payload["summary"].__setitem__(
                "seeds", list(reversed(payload["summary"]["seeds"]))
            ),
            "wrong ids": lambda payload: payload["summary"].__setitem__(
                "sandbox_ids", list(reversed(payload["summary"]["sandbox_ids"]))
            ),
            "wrong point count": lambda payload: payload["summary"].__setitem__(
                "total_trajectory_points",
                payload["summary"]["total_trajectory_points"] + 1,
            ),
            "wrong concurrency": lambda payload: payload["summary"].__setitem__(
                "concurrent", False
            ),
            "unknown cleanup": lambda payload: payload["summary"].__setitem__(
                "cleanup", "unknown"
            ),
            "invalid action index": lambda payload: payload["results"][0].__setitem__(
                "actions", [999]
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = deepcopy(self._raw_generation())
                mutate(payload)
                attempts = load_rollout_trails(
                    self._write(payload, f"raw-{label.replace(' ', '-')}.json")
                )
                self.assertFalse(any(attempt.daytona_verified for attempt in attempts))

    def test_loaded_generic_recording_never_becomes_local_preview(self) -> None:
        payload = {
            "seed": 7,
            "trajectory": [{"x": 10.0, "y": 20.0}, {"x": 12.0, "y": 24.0}],
        }

        attempts = load_rollout_trails(self._write(payload, "generic-recording.json"))

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].provenance, "UNVERIFIED RECORDED REPLAY")
        self.assertFalse(attempts[0].daytona_verified)
        with self.assertRaisesRegex(ValueError, "no recorded universe"):
            _replay_environment(attempts[0])

    def test_cli_requires_explicit_real_replay_or_local_preview(self) -> None:
        for arguments in ([], ["--local-preview", "--generation", "2"]):
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [sys.executable, "visual_demo.py", *arguments],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2)

    def test_daytona_worker_bundle_has_no_pygame_dependency(self) -> None:
        for relative in ("rollout_worker.py", "daytona_worker_entry.py"):
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            imported = {
                node.names[0].name.split(".")[0]
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Import) and node.names
            }
            imported.update(
                node.module.split(".")[0]
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.ImportFrom) and node.module
            )
            self.assertNotIn("pygame", imported)
        requirements = (REPO_ROOT / "requirements-daytona.txt").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("pygame", requirements.lower())


if __name__ == "__main__":
    unittest.main()
