"""Unit tests for controller-side REINFORCE training."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    from torch import nn

    import trainer
    from gravity_env import (
        ACTION_HOLD_STEPS,
        OBSERVATION_DIM,
        GravityEnv,
        action_to_vector,
    )
    from rl_policy import create_policy, encode_policy_weights
    from rollout_worker import execute_job
except ImportError:  # The project requirements may not yet be installed.
    torch = None
    nn = None
    trainer = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RewardToGoTests(unittest.TestCase):
    def test_reward_to_go_uses_gamma_and_episode_boundary(self) -> None:
        actual = trainer.compute_reward_to_go([1.0, 2.0, 3.0], gamma=0.5)
        expected = torch.tensor([2.75, 3.5, 3.0])
        self.assertTrue(torch.allclose(actual, expected))

    def test_default_gamma_is_point_nine_nine(self) -> None:
        actual = trainer.compute_reward_to_go([0.0, 1.0])
        self.assertTrue(torch.allclose(actual, torch.tensor([0.99, 1.0])))

    def test_invalid_gamma_and_non_finite_rewards_are_rejected(self) -> None:
        for invalid_gamma in (-0.1, 1.01, math.inf, math.nan):
            with self.subTest(gamma=invalid_gamma):
                with self.assertRaises(ValueError):
                    trainer.compute_reward_to_go([1.0], invalid_gamma)
        with self.assertRaises(ValueError):
            trainer.compute_reward_to_go([float("nan")])

    def test_normalization_is_stable_for_constant_and_singleton_batches(self) -> None:
        constant = trainer.normalize_advantages(torch.tensor([2.0, 2.0, 2.0]))
        singleton = trainer.normalize_advantages(torch.tensor([2.0]))
        self.assertTrue(torch.equal(constant, torch.zeros(3)))
        self.assertTrue(torch.equal(singleton, torch.tensor([2.0])))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ReinforceUpdateTests(unittest.TestCase):
    OBS_DIM = 3

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = nn.Sequential(
            nn.Linear(self.OBS_DIM, 8),
            nn.Tanh(),
            nn.Linear(8, 9),
        )
        self.optimizer = trainer.create_optimizer(self.model, learning_rate=0.02)

    @staticmethod
    def _rollouts() -> list[dict[str, object]]:
        return [
            {
                "observations": [[1.0, 0.0, 0.2], [0.8, 0.1, 0.3]],
                "actions": [1, 2],
                "rewards": [0.2, 2.0],
                "reward": 2.2,
                "success": True,
                "termination": "portal",
                "min_clearance": 15.0,
            },
            {
                "observations": [[-1.0, 0.5, 0.0], [-0.4, 0.8, -0.2]],
                "actions": [3, 3],
                "rewards": [-0.1, -2.0],
                "reward": -2.1,
                "success": False,
                "termination": "collision_planet",
                "min_clearance": -0.5,
            },
        ]

    def test_update_changes_weights_and_returns_finite_metrics(self) -> None:
        before = [parameter.detach().clone() for parameter in self.model.parameters()]

        metrics = trainer.reinforce_update(
            self.model,
            self.optimizer,
            self._rollouts(),
        )

        after = list(self.model.parameters())
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, after)))
        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["transitions"], 4)
        for key in (
            "loss",
            "policy_loss",
            "entropy",
            "gradient_norm",
            "mean_reward_to_go",
            "std_reward_to_go",
            "advantage_mean",
            "advantage_std",
        ):
            self.assertTrue(math.isfinite(metrics[key]), key)
        self.assertGreater(metrics["entropy"], 0.0)
        self.assertLessEqual(metrics["gradient_norm"], 100.0)

    def test_empty_batch_is_rejected_without_changing_weights(self) -> None:
        before = [parameter.detach().clone() for parameter in self.model.parameters()]
        with self.assertRaisesRegex(trainer.RolloutValidationError, "must not be empty"):
            trainer.reinforce_update(self.model, self.optimizer, [])
        self.assertTrue(
            all(torch.equal(old, new) for old, new in zip(before, self.model.parameters()))
        )

    def test_empty_episode_is_rejected(self) -> None:
        rollout = {"observations": [], "actions": [], "rewards": []}
        with self.assertRaisesRegex(trainer.RolloutValidationError, "must have shape|empty"):
            trainer.reinforce_update(self.model, self.optimizer, [rollout])

    def test_missing_field_is_rejected(self) -> None:
        rollout = {"observations": [[0.0, 0.0, 0.0]], "actions": [0]}
        with self.assertRaisesRegex(trainer.RolloutValidationError, "missing.*rewards"):
            trainer.reinforce_update(self.model, self.optimizer, [rollout])

    def test_mismatched_lengths_are_rejected(self) -> None:
        rollout = {
            "observations": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "actions": [0],
            "rewards": [1.0, 1.0],
        }
        with self.assertRaisesRegex(trainer.RolloutValidationError, "lengths differ"):
            trainer.reinforce_update(self.model, self.optimizer, [rollout])

    def test_fractional_out_of_range_and_non_finite_actions_are_rejected(self) -> None:
        for action in (1.5, -1, 9, float("nan")):
            rollout = {
                "observations": [[0.0, 0.0, 0.0]],
                "actions": [action],
                "rewards": [1.0],
            }
            with self.subTest(action=action):
                with self.assertRaises(trainer.RolloutValidationError):
                    trainer.reinforce_update(self.model, self.optimizer, [rollout])

    def test_policy_must_return_nine_logits(self) -> None:
        invalid_model = nn.Linear(self.OBS_DIM, 8)
        optimizer = trainer.create_optimizer(invalid_model)
        with self.assertRaisesRegex(ValueError, "9"):
            trainer.reinforce_update(invalid_model, optimizer, self._rollouts())

    def test_gradient_is_clipped_before_optimizer_step(self) -> None:
        metrics = trainer.reinforce_update(
            self.model,
            self.optimizer,
            self._rollouts(),
            max_grad_norm=0.01,
        )
        # clip_grad_norm_ reports the pre-clipping norm; inspect actual gradients.
        actual_norm = torch.linalg.vector_norm(
            torch.cat(
                [
                    parameter.grad.detach().flatten()
                    for parameter in self.model.parameters()
                    if parameter.grad is not None
                ]
            )
        ).item()
        self.assertLessEqual(actual_norm, 0.010001)
        self.assertGreaterEqual(metrics["gradient_norm"], actual_norm)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GenerationMetricsTests(unittest.TestCase):
    def test_requested_metrics_are_computed(self) -> None:
        metrics = trainer.summarize_generation(
            [
                {
                    "reward": 10.0,
                    "success": True,
                    "termination": "portal",
                    "min_clearance": 8.0,
                },
                {
                    "rewards": [-1.0, -2.0],
                    "success": False,
                    "termination": "collision_asteroid",
                    "info": {"min_clearance": -1.0},
                },
            ]
        )
        self.assertEqual(metrics["worlds"], 2)
        self.assertAlmostEqual(metrics["average_reward"], 3.5)
        self.assertEqual(metrics["best_reward"], 10.0)
        self.assertEqual(metrics["success_rate"], 0.5)
        self.assertEqual(metrics["collision_rate"], 0.5)
        self.assertAlmostEqual(metrics["average_min_clearance"], 3.5)

    def test_missing_clearance_is_reported_as_none(self) -> None:
        metrics = trainer.summarize_generation([{"reward": 1.0}])
        self.assertIsNone(metrics["average_min_clearance"])

    def test_seed_batches_are_unique_and_reproducible(self) -> None:
        first = trainer.generation_seeds(123, generation=4, worlds=8)
        second = trainer.generation_seeds(123, generation=4, worlds=8)
        other_generation = trainer.generation_seeds(123, generation=5, worlds=8)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertNotEqual(first, other_generation)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EnvironmentLearningContractTests(unittest.TestCase):
    def test_observation_dimension_is_fixed_across_seeded_universes(self) -> None:
        dimensions = {
            len(GravityEnv(seed=seed, max_steps=20).get_observation())
            for seed in range(24)
        }
        self.assertEqual(dimensions, {OBSERVATION_DIM})
        self.assertEqual(OBSERVATION_DIM, 44)

    def test_minimum_clearance_uses_body_and_ship_surfaces(self) -> None:
        env = GravityEnv(seed=1, max_steps=20)
        env.ship_position = [100.0, 100.0]
        env.planets = [{"position": [160.0, 100.0], "radius": 20.0}]
        env.asteroids = [{"position": [100.0, 170.0], "radius": 5.0}]

        self.assertEqual(env.surface_clearances(), [30.0, 55.0])
        self.assertEqual(env.minimum_clearance(), 30.0)

    def test_danger_penalty_grows_as_surface_clearance_shrinks(self) -> None:
        env = GravityEnv(seed=2, max_steps=20)
        env.planets = [{"position": [600.0, 400.0], "radius": 40.0}]
        env.asteroids = []

        env.ship_position = [470.0, 400.0]
        far_reward = env._shaped_reward(env.distance_to_target(), [0.0, 0.0])

        env.ship_position = [521.0, 400.0]
        near_clearance = env.minimum_clearance()
        near_reward = env._shaped_reward(env.distance_to_target(), [0.0, 0.0])

        self.assertAlmostEqual(near_clearance, env.SAFE_MARGIN / 2.0)
        self.assertLess(near_reward, far_reward)
        self.assertAlmostEqual(
            far_reward - near_reward,
            env.SAFETY_SCALE * 0.5**2,
        )

    def test_collision_penalty_is_applied_to_shaped_reward(self) -> None:
        env = GravityEnv(seed=3, max_steps=20)
        env.planets = []
        env.asteroids = []
        env.success = False
        env.collision = {"kind": "planet", "index": 0}
        env.status = "collision_planet"

        reward = env._shaped_reward(env.distance_to_target(), [0.0, 0.0])

        self.assertAlmostEqual(reward, env.STEP_COST + env.COLLISION_PENALTY)
        self.assertEqual(env._event_reward(), env.COLLISION_PENALTY)

    def test_portal_bonus_is_applied_to_shaped_reward(self) -> None:
        env = GravityEnv(seed=4, max_steps=20)
        env.planets = []
        env.asteroids = []
        env.success = True
        env.collision = None
        env.status = "success"

        reward = env._shaped_reward(env.distance_to_target(), [0.0, 0.0])

        self.assertAlmostEqual(reward, env.STEP_COST + env.PORTAL_BONUS)
        self.assertEqual(env._event_reward(), env.PORTAL_BONUS)

    def test_held_action_matches_repeated_real_physics_steps_deterministically(self) -> None:
        held_env = GravityEnv(seed=18_473, max_steps=100)
        repeated_env = GravityEnv(seed=18_473, max_steps=100)
        action_index = 3
        action_vector = action_to_vector(action_index)

        held_observation, held_reward, terminated, truncated, info = (
            held_env.step_discrete(action_index, hold_steps=ACTION_HOLD_STEPS)
        )
        repeated_reward = 0.0
        for _ in range(ACTION_HOLD_STEPS):
            _, reward, repeated_terminated, repeated_truncated, _ = repeated_env.step(
                action_vector
            )
            repeated_reward += reward
            if repeated_terminated or repeated_truncated:
                break

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["physics_steps_held"], ACTION_HOLD_STEPS)
        self.assertEqual(held_env.timestep, ACTION_HOLD_STEPS)
        self.assertEqual(held_observation, repeated_env.get_observation())
        self.assertEqual(held_env.trajectory, repeated_env.trajectory)
        self.assertAlmostEqual(held_reward, repeated_reward)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class WorkerTrainingContractTests(unittest.TestCase):
    @staticmethod
    def _job(policy_weights: str | None, *, max_steps: int = 12) -> dict[str, object]:
        return {
            "seed": 18_473,
            "policy_version": 0 if policy_weights is None else 7,
            "policy_weights": policy_weights,
            "max_steps": max_steps,
        }

    def test_worker_exposes_aligned_training_sequences_and_local_sandbox_key(
        self,
    ) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            result = execute_job(self._job(None))

        self.assertIn("sandbox_id", result)
        self.assertIsNone(result["sandbox_id"])
        self.assertIn("observations", result)
        self.assertEqual(result["steps"], len(result["observations"]))
        self.assertEqual(result["steps"], len(result["actions"]))
        self.assertEqual(result["steps"], len(result["action_vectors"]))
        self.assertEqual(result["steps"], len(result["rewards"]))
        self.assertTrue(result["observations"])
        self.assertTrue(
            all(len(observation) == OBSERVATION_DIM for observation in result["observations"])
        )
        for action, vector in zip(result["actions"], result["action_vectors"]):
            expected_vector = action_to_vector(action)
            self.assertAlmostEqual(vector[0], expected_vector[0])
            self.assertAlmostEqual(vector[1], expected_vector[1])

    def test_worker_action_hold_uses_fixed_physics_substeps(self) -> None:
        result = execute_job(self._job(None, max_steps=3))

        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["physics_steps"], 3 * ACTION_HOLD_STEPS)
        self.assertEqual(len(result["trajectory"]), result["physics_steps"] + 1)

    def test_seeded_worker_is_reproducible_for_null_and_neural_weights(self) -> None:
        model = create_policy(OBSERVATION_DIM, seed=55)
        encoded_weights = encode_policy_weights(model)

        for weights in (None, encoded_weights):
            with self.subTest(mode="random_v0" if weights is None else "neural"):
                job = self._job(weights)
                first = execute_job(job)
                second = execute_job(job)
                self.assertEqual(first, second)

    def test_worker_rejects_mislabeled_policy_payloads(self) -> None:
        encoded_weights = encode_policy_weights(create_policy(OBSERVATION_DIM, seed=55))
        invalid_jobs = (
            {
                "seed": 1,
                "policy_version": 0,
                "policy_weights": encoded_weights,
                "max_steps": 2,
            },
            {
                "seed": 1,
                "policy_version": 1,
                "policy_weights": None,
                "max_steps": 2,
            },
        )
        for job in invalid_jobs:
            with self.subTest(policy_version=job["policy_version"]):
                with self.assertRaises(ValueError):
                    execute_job(job)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DaytonaTrainingBoundaryTests(unittest.TestCase):
    def test_policy_v0_controller_distribution_is_uniform(self) -> None:
        model = create_policy(3, seed=123)
        trainer.initialize_uniform_policy_v0(model)
        probabilities = torch.softmax(model([0.1, 0.2, 0.3]), dim=-1)
        self.assertTrue(torch.allclose(probabilities, torch.full((9,), 1.0 / 9.0)))

    def test_missing_daytona_module_has_no_local_fallback(self) -> None:
        with mock.patch.dict("sys.modules", {"daytona_orchestrator": None}):
            with self.assertRaisesRegex(RuntimeError, "no local rollout fallback"):
                trainer._load_daytona_run_generation()

    def test_async_loop_passes_encoded_weights_and_saves_checkpoints(self) -> None:
        try:
            import rl_policy  # noqa: F401
        except ImportError:
            self.skipTest("rl_policy integration is not available yet")

        calls: list[dict[str, object]] = []

        async def fake_daytona_generation(**kwargs: object) -> list[dict[str, object]]:
            calls.append(kwargs)
            return [
                {
                    "observations": [
                        [0.1, 0.2, 0.3],
                        [0.2, 0.1, 0.4],
                    ],
                    "actions": [world_index % 9, (world_index + 1) % 9],
                    "rewards": [-0.1, float(world_index + 1)],
                    "reward": float(world_index + 0.9),
                    "success": world_index == 0,
                    "termination": "portal" if world_index == 0 else "timeout",
                    "min_clearance": float(world_index + 2),
                }
                for world_index in range(len(kwargs["seeds"]))
            ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = io.StringIO()
            with (
                mock.patch.object(
                    trainer,
                    "_load_daytona_run_generation",
                    return_value=fake_daytona_generation,
                ),
                contextlib.redirect_stdout(output),
            ):
                history = asyncio.run(
                    trainer.run_daytona_training(
                        generations=2,
                        worlds=2,
                        max_steps=5,
                        obs_dim=3,
                        base_seed=99,
                        checkpoint_dir=temporary_directory,
                    )
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["policy_version"], 0)
            self.assertEqual(calls[0]["max_steps"], 5)
            self.assertEqual(len(calls[0]["seeds"]), 2)
            self.assertIsNone(calls[0]["policy_weights"])
            self.assertEqual(calls[1]["policy_version"], 1)
            self.assertIsInstance(calls[1]["policy_weights"], str)
            self.assertTrue(calls[1]["policy_weights"])
            self.assertEqual(history[0]["next_policy_version"], 1)
            self.assertEqual(history[1]["next_policy_version"], 2)
            self.assertTrue((Path(temporary_directory) / "policy_v000.pt").is_file())
            self.assertTrue((Path(temporary_directory) / "policy_v001.pt").is_file())
            self.assertTrue((Path(temporary_directory) / "policy_v002.pt").is_file())
            rendered = output.getvalue()
            self.assertIn("GENERATION 00", rendered)
            self.assertIn("GENERATION 01", rendered)
            self.assertIn("Daytona worlds: 2", rendered)
            self.assertIn("Policy updated → v2", rendered)


class VisualTrajectoryIntegrityTests(unittest.TestCase):
    def test_rollout_loader_preserves_world_and_training_identity(self) -> None:
        from visual_demo import load_rollout_trails

        payload = {
            "generation": 3,
            "policy_version": 5,
            "rollouts": [
                {
                    "seed": 18473,
                    "sandbox_id": "real-sandbox",
                    "reward": 12.5,
                    "success": True,
                    "trajectory": [
                        {"x": 10.0, "y": 20.0},
                        {"x": 15.0, "y": 24.0},
                    ],
                },
                {
                    "seed": 18474,
                    "trajectory": [
                        {"x": 11.0, "y": 21.0},
                        {"x": 16.0, "y": 25.0},
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            attempts = load_rollout_trails(path)

        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0].seed, 18473)
        self.assertEqual(attempts[0].generation, 3)
        self.assertEqual(attempts[0].policy_version, 5)
        self.assertEqual(attempts[0].sandbox_id, "real-sandbox")
        self.assertEqual(attempts[0].reward, 12.5)
        self.assertIsNone(attempts[1].reward)
        self.assertIsNone(attempts[1].success)
        self.assertIsNone(attempts[1].sandbox_id)

    def test_manual_attempt_keeps_seed_without_claiming_a_policy(self) -> None:
        from visual_demo import _completed_attempt

        env = GravityEnv(seed=91)
        env.step((0.0, 0.0))
        attempt = _completed_attempt(env)

        self.assertIsNotNone(attempt)
        assert attempt is not None
        self.assertEqual(attempt.seed, 91)
        self.assertIsNone(attempt.policy_version)
        self.assertIsNone(attempt.generation)
        self.assertIsNone(attempt.sandbox_id)


if __name__ == "__main__":
    unittest.main()
