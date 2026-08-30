"""Small categorical flight policy and safe JSON weight transport.

The policy deliberately knows nothing about the physics environment or the
Daytona execution layer.  It maps a fixed-size numerical observation to nine
raw action logits.  Callers own the observation construction, the environment
step, and the lifetime of the dedicated sampling generator.

Weights are transported as a base64-encoded, canonical JSON document.  Each
tensor is represented by explicit metadata and base64-encoded little-endian
float32 bytes.  This module never uses ``torch.save``, pickle, or executable
deserialization formats.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import struct
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from gravity_env import ACTION_VECTORS


NUM_ACTIONS = len(ACTION_VECTORS)
HIDDEN_SIZE = 64
SERIALIZATION_FORMAT = "gravity-gauntlet-policy-state"
SERIALIZATION_VERSION = 1
MAX_ENCODED_PAYLOAD_BYTES = 64 * 1024 * 1024

class PolicyNetwork(nn.Module):
    """A tiny ``obs_dim -> 64 -> 64 -> 9`` categorical policy.

    ``forward`` returns raw logits.  Softmax belongs at the sampling/loss
    boundary so training can use numerically stable categorical log
    probabilities.
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.obs_dim = _validate_obs_dim(obs_dim)
        self.fc1 = nn.Linear(self.obs_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.action_head = nn.Linear(HIDDEN_SIZE, NUM_ACTIONS)

    def forward(self, observation: Tensor | Sequence[float]) -> Tensor:
        parameter = self.fc1.weight
        values = torch.as_tensor(
            observation,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        if values.ndim < 1 or values.shape[-1] != self.obs_dim:
            actual = tuple(values.shape)
            raise ValueError(
                "observation's final dimension must be "
                f"{self.obs_dim}, received shape {actual}"
            )

        hidden = torch.tanh(self.fc1(values))
        hidden = torch.tanh(self.fc2(hidden))
        return self.action_head(hidden)


def create_policy(obs_dim: int, *, seed: int | None = None) -> PolicyNetwork:
    """Create a CPU float32 policy, optionally with reproducible weights.

    Seeded construction is isolated with ``fork_rng`` so it does not consume
    or replace the caller's process-global PyTorch RNG state.
    """

    validated_obs_dim = _validate_obs_dim(obs_dim)
    if seed is None:
        return PolicyNetwork(validated_obs_dim).cpu().float()

    validated_seed = _validate_integer(seed, "seed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_torch_seed(validated_seed))
        return PolicyNetwork(validated_obs_dim).cpu().float()


def action_probabilities(
    model: PolicyNetwork,
    observation: Tensor | Sequence[float],
) -> Tensor:
    """Return softmax-normalized action probabilities."""

    logits = model(observation)
    probabilities = torch.softmax(logits, dim=-1)
    if not bool(torch.isfinite(probabilities).all().item()):
        raise ValueError("policy produced non-finite action probabilities")
    return probabilities


def action_index_to_vector(action_index: int) -> tuple[float, float]:
    """Map a categorical action index to a unit-length continuous thrust."""

    index = _validate_integer(action_index, "action_index")
    if not 0 <= index < NUM_ACTIONS:
        raise ValueError(f"action_index must be between 0 and {NUM_ACTIONS - 1}")
    return ACTION_VECTORS[index]


def make_action_generator(
    universe_seed: int,
    policy_version: int = 0,
) -> torch.Generator:
    """Create the dedicated CPU RNG for one rollout's policy decisions.

    The stable SHA-256 derivation avoids Python's process-randomized ``hash``
    and keeps action sampling reproducible for a universe/policy pair.
    """

    universe_seed = _validate_integer(universe_seed, "universe_seed")
    policy_version = _validate_integer(policy_version, "policy_version")
    seed_material = (
        f"gravity-gauntlet/action-sampling/v1/{universe_seed}/{policy_version}"
    ).encode("ascii")
    digest = hashlib.sha256(seed_material).digest()
    derived_seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed)
    return generator


def sample_action(
    model: PolicyNetwork,
    observation: Tensor | Sequence[float],
    generator: torch.Generator,
) -> int:
    """Stochastically sample one learned-policy action reproducibly.

    A caller must pass the dedicated generator created for the rollout.  The
    explicit argument prevents accidental dependence on global RNG state.
    """

    _validate_cpu_generator(generator)
    with torch.no_grad():
        probabilities = action_probabilities(model, observation)
    if probabilities.ndim == 2 and probabilities.shape[0] == 1:
        probabilities = probabilities.squeeze(0)
    if probabilities.ndim != 1 or probabilities.shape[0] != NUM_ACTIONS:
        raise ValueError("sample_action expects exactly one observation")

    sampled = torch.multinomial(
        probabilities.detach().to(device="cpu", dtype=torch.float32),
        num_samples=1,
        replacement=True,
        generator=generator,
    )
    return int(sampled.item())


def sample_random_action(generator: torch.Generator) -> int:
    """Sample one uniform policy-v0 action from a dedicated generator."""

    _validate_cpu_generator(generator)
    sampled = torch.randint(
        low=0,
        high=NUM_ACTIONS,
        size=(1,),
        generator=generator,
        device="cpu",
    )
    return int(sampled.item())


def encode_policy_weights(model: PolicyNetwork) -> str:
    """Encode a policy state dict as deterministic, non-pickle base64 text."""

    if not isinstance(model, PolicyNetwork):
        raise TypeError("model must be a PolicyNetwork")

    tensor_records: list[dict[str, Any]] = []
    for name, tensor in sorted(model.state_dict().items()):
        cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
        cpu_tensor = cpu_tensor.contiguous()
        if not bool(torch.isfinite(cpu_tensor).all().item()):
            raise ValueError(f"state tensor {name!r} contains non-finite values")

        flat_values = cpu_tensor.reshape(-1).tolist()
        raw_bytes = struct.pack(f"<{len(flat_values)}f", *flat_values)
        tensor_records.append(
            {
                "data": base64.b64encode(raw_bytes).decode("ascii"),
                "dtype": "float32-le",
                "name": name,
                "shape": list(cpu_tensor.shape),
            }
        )

    envelope = {
        "architecture": {
            "activation": "tanh",
            "hidden_sizes": [HIDDEN_SIZE, HIDDEN_SIZE],
            "num_actions": NUM_ACTIONS,
            "obs_dim": model.obs_dim,
        },
        "format": SERIALIZATION_FORMAT,
        "state_dict": tensor_records,
        "version": SERIALIZATION_VERSION,
    }
    canonical_json = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return base64.b64encode(canonical_json).decode("ascii")


def decode_policy_weights(payload: str, obs_dim: int) -> PolicyNetwork:
    """Safely reconstruct a CPU float32 policy from ``encode`` output.

    The decoder accepts data only.  It strictly checks the versioned envelope,
    architecture, expected state-dict names, shapes, dtypes, byte counts, and
    finite values before loading tensors into the known local architecture.
    """

    validated_obs_dim = _validate_obs_dim(obs_dim)
    if not isinstance(payload, str) or not payload:
        raise ValueError("policy weight payload must be a non-empty base64 string")
    if len(payload) > MAX_ENCODED_PAYLOAD_BYTES:
        raise ValueError("policy weight payload exceeds the safe size limit")

    try:
        document_bytes = base64.b64decode(payload.encode("ascii"), validate=True)
        envelope = json.loads(document_bytes.decode("ascii"))
    except (
        UnicodeEncodeError,
        UnicodeDecodeError,
        binascii.Error,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("invalid policy weight payload") from exc

    if not isinstance(envelope, dict):
        raise ValueError("policy weight envelope must be a JSON object")
    if envelope.get("format") != SERIALIZATION_FORMAT:
        raise ValueError("unsupported policy weight format")
    if envelope.get("version") != SERIALIZATION_VERSION:
        raise ValueError("unsupported policy weight version")

    expected_architecture = {
        "activation": "tanh",
        "hidden_sizes": [HIDDEN_SIZE, HIDDEN_SIZE],
        "num_actions": NUM_ACTIONS,
        "obs_dim": validated_obs_dim,
    }
    if envelope.get("architecture") != expected_architecture:
        raise ValueError("policy architecture does not match obs_dim and v1 schema")

    # A fixed seed prevents decoder construction from changing global RNG state;
    # every initialized value is replaced by strictly validated decoded data.
    model = create_policy(validated_obs_dim, seed=0)
    expected_state = model.state_dict()
    records = envelope.get("state_dict")
    if not isinstance(records, list):
        raise ValueError("policy state_dict must be a list")

    records_by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError("invalid tensor record")
        name = record["name"]
        if name in records_by_name:
            raise ValueError(f"duplicate tensor record {name!r}")
        records_by_name[name] = record

    if set(records_by_name) != set(expected_state):
        raise ValueError("policy state_dict keys do not match the expected model")

    decoded_state: dict[str, Tensor] = {}
    for name, expected_tensor in expected_state.items():
        record = records_by_name[name]
        if record.get("dtype") != "float32-le":
            raise ValueError(f"tensor {name!r} must use float32-le encoding")

        shape = _decode_shape(record.get("shape"), name)
        expected_shape = tuple(expected_tensor.shape)
        if shape != expected_shape:
            raise ValueError(
                f"tensor {name!r} has shape {shape}, expected {expected_shape}"
            )

        data = record.get("data")
        if not isinstance(data, str):
            raise ValueError(f"tensor {name!r} has no base64 data")
        try:
            raw_bytes = base64.b64decode(data.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error) as exc:
            raise ValueError(f"tensor {name!r} contains invalid base64 data") from exc

        element_count = expected_tensor.numel()
        expected_byte_count = element_count * 4
        if len(raw_bytes) != expected_byte_count:
            raise ValueError(
                f"tensor {name!r} contains {len(raw_bytes)} bytes; "
                f"expected {expected_byte_count}"
            )

        values = struct.unpack(f"<{element_count}f", raw_bytes)
        decoded_tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
        if not bool(torch.isfinite(decoded_tensor).all().item()):
            raise ValueError(f"tensor {name!r} contains non-finite values")
        decoded_state[name] = decoded_tensor

    model.load_state_dict(decoded_state, strict=True)
    model.cpu().float()
    model.eval()
    return model


def _validate_obs_dim(obs_dim: int) -> int:
    value = _validate_integer(obs_dim, "obs_dim")
    if value <= 0:
        raise ValueError("obs_dim must be a positive integer")
    return value


def _validate_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _torch_seed(value: int) -> int:
    # PyTorch accepts signed 64-bit seed semantics.  Normalizing also supports
    # arbitrary-size Python integers without platform-dependent overflow.
    return value % (1 << 63)


def _validate_cpu_generator(generator: torch.Generator) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    if str(generator.device) != "cpu":
        raise ValueError("action sampling requires a CPU torch.Generator")


def _decode_shape(value: Any, tensor_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"tensor {tensor_name!r} has an invalid shape")
    dimensions: list[int] = []
    for dimension in value:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
        ):
            raise ValueError(f"tensor {tensor_name!r} has an invalid shape")
        dimensions.append(dimension)
    return tuple(dimensions)


__all__ = [
    "ACTION_VECTORS",
    "HIDDEN_SIZE",
    "NUM_ACTIONS",
    "PolicyNetwork",
    "action_index_to_vector",
    "action_probabilities",
    "create_policy",
    "decode_policy_weights",
    "encode_policy_weights",
    "make_action_generator",
    "sample_action",
    "sample_random_action",
]
