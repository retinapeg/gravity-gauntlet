"""Tests for the tiny categorical Gravity Gauntlet policy."""

from __future__ import annotations

import base64
import json
import math
import unittest

import torch

from gravity_env import OBSERVATION_DIM
from rl_policy import (
    ACTION_VECTORS,
    NUM_ACTIONS,
    PolicyNetwork,
    action_index_to_vector,
    action_probabilities,
    create_policy,
    decode_policy_weights,
    encode_policy_weights,
    make_action_generator,
    sample_action,
    sample_random_action,
)


OBS_DIM = OBSERVATION_DIM


class PolicyArchitectureTests(unittest.TestCase):
    def test_policy_has_required_architecture_and_output_shapes(self) -> None:
        policy = create_policy(OBS_DIM, seed=17)

        self.assertIsInstance(policy, PolicyNetwork)
        self.assertEqual(policy.fc1.in_features, OBS_DIM)
        self.assertEqual(policy.fc1.out_features, 64)
        self.assertEqual(policy.fc2.in_features, 64)
        self.assertEqual(policy.fc2.out_features, 64)
        self.assertEqual(policy.action_head.in_features, 64)
        self.assertEqual(policy.action_head.out_features, NUM_ACTIONS)
        self.assertEqual(tuple(policy(torch.zeros(OBS_DIM)).shape), (9,))
        self.assertEqual(tuple(policy(torch.zeros(5, OBS_DIM)).shape), (5, 9))

    def test_probability_normalization(self) -> None:
        policy = create_policy(OBS_DIM, seed=23)
        probabilities = action_probabilities(policy, torch.randn(7, OBS_DIM))

        self.assertTrue(bool(torch.isfinite(probabilities).all()))
        self.assertTrue(bool((probabilities >= 0.0).all()))
        self.assertTrue(bool((probabilities <= 1.0).all()))
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones(7),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_seeded_construction_is_reproducible(self) -> None:
        first = create_policy(OBS_DIM, seed=91)
        second = create_policy(OBS_DIM, seed=91)
        third = create_policy(OBS_DIM, seed=92)

        for first_tensor, second_tensor in zip(
            first.state_dict().values(), second.state_dict().values()
        ):
            self.assertTrue(torch.equal(first_tensor, second_tensor))
        self.assertTrue(
            any(
                not torch.equal(first_tensor, third_tensor)
                for first_tensor, third_tensor in zip(
                    first.state_dict().values(), third.state_dict().values()
                )
            )
        )

    def test_invalid_observation_shape_is_rejected(self) -> None:
        policy = create_policy(OBS_DIM, seed=1)
        with self.assertRaises(ValueError):
            policy(torch.zeros(OBS_DIM - 1))


class ActionTests(unittest.TestCase):
    def test_all_action_indices_map_to_expected_thrust_vectors(self) -> None:
        diagonal = 1.0 / math.sqrt(2.0)
        expected_vectors = (
            (0.0, 0.0),
            (0.0, -1.0),
            (diagonal, -diagonal),
            (1.0, 0.0),
            (diagonal, diagonal),
            (0.0, 1.0),
            (-diagonal, diagonal),
            (-1.0, 0.0),
            (-diagonal, -diagonal),
        )

        self.assertEqual(len(ACTION_VECTORS), NUM_ACTIONS)
        self.assertEqual(ACTION_VECTORS, expected_vectors)

        for action_index in range(NUM_ACTIONS):
            vector = action_index_to_vector(action_index)
            self.assertEqual(vector, ACTION_VECTORS[action_index])
            self.assertTrue(all(-1.0 <= component <= 1.0 for component in vector))
            expected_magnitude = 0.0 if action_index == 0 else 1.0
            self.assertAlmostEqual(math.hypot(*vector), expected_magnitude)

        with self.assertRaises(ValueError):
            action_index_to_vector(-1)
        with self.assertRaises(ValueError):
            action_index_to_vector(NUM_ACTIONS)

    def test_seeded_learned_policy_sampling_is_reproducible(self) -> None:
        policy = create_policy(OBS_DIM, seed=5)
        observation = torch.linspace(-1.0, 1.0, OBS_DIM)
        first_generator = make_action_generator(18473, 7)
        second_generator = make_action_generator(18473, 7)

        first_actions = [
            sample_action(policy, observation, generator=first_generator)
            for _ in range(50)
        ]
        second_actions = [
            sample_action(policy, observation, generator=second_generator)
            for _ in range(50)
        ]
        self.assertEqual(first_actions, second_actions)

    def test_seeded_policy_v0_sampling_is_reproducible(self) -> None:
        first_generator = make_action_generator(18473, 0)
        second_generator = make_action_generator(18473, 0)
        different_generator = make_action_generator(18473, 1)

        first_actions = [sample_random_action(first_generator) for _ in range(50)]
        second_actions = [sample_random_action(second_generator) for _ in range(50)]
        different_actions = [
            sample_random_action(different_generator) for _ in range(50)
        ]

        self.assertEqual(first_actions, second_actions)
        self.assertNotEqual(first_actions, different_actions)


class SerializationTests(unittest.TestCase):
    def test_weight_roundtrip_preserves_every_tensor_and_logits(self) -> None:
        original = create_policy(OBS_DIM, seed=314159)
        observation = torch.randn(11, OBS_DIM)

        payload = encode_policy_weights(original)
        restored = decode_policy_weights(payload, OBS_DIM)

        self.assertIsInstance(payload, str)
        self.assertEqual(restored.obs_dim, OBS_DIM)
        self.assertEqual(next(restored.parameters()).device.type, "cpu")
        self.assertEqual(next(restored.parameters()).dtype, torch.float32)
        for name, original_tensor in original.state_dict().items():
            self.assertTrue(
                torch.equal(original_tensor, restored.state_dict()[name]),
                msg=name,
            )
        torch.testing.assert_close(
            original(observation),
            restored(observation),
            rtol=0.0,
            atol=0.0,
        )

    def test_serialization_is_canonical_json_and_deterministic(self) -> None:
        policy = create_policy(OBS_DIM, seed=2718)
        first_payload = encode_policy_weights(policy)
        second_payload = encode_policy_weights(policy)
        envelope = json.loads(base64.b64decode(first_payload).decode("ascii"))

        self.assertEqual(first_payload, second_payload)
        self.assertEqual(envelope["format"], "gravity-gauntlet-policy-state")
        self.assertEqual(envelope["version"], 1)
        self.assertEqual(envelope["architecture"]["obs_dim"], OBS_DIM)
        self.assertTrue(envelope["state_dict"])
        self.assertTrue(
            all(
                record["dtype"] == "float32-le"
                for record in envelope["state_dict"]
            )
        )
        self.assertEqual(
            first_payload,
            encode_policy_weights(decode_policy_weights(first_payload, OBS_DIM)),
        )

    def test_decoder_rejects_invalid_or_mismatched_payloads(self) -> None:
        payload = encode_policy_weights(create_policy(OBS_DIM, seed=8))

        with self.assertRaises(ValueError):
            decode_policy_weights("not base64!", OBS_DIM)
        with self.assertRaises(ValueError):
            decode_policy_weights(payload, OBS_DIM + 1)

        envelope = json.loads(base64.b64decode(payload).decode("ascii"))
        envelope["version"] = 999
        unsupported_payload = base64.b64encode(
            json.dumps(envelope).encode("ascii")
        ).decode("ascii")
        with self.assertRaises(ValueError):
            decode_policy_weights(unsupported_payload, OBS_DIM)


if __name__ == "__main__":
    unittest.main()
