"""Controller-side REINFORCE training for Gravity Gauntlet.

This module deliberately does not import or execute the physics environment or
the rollout worker.  The command-line training pathway obtains every episode
from :func:`daytona_orchestrator.run_generation`; only gradient calculation and
checkpointing happen in this controller process.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import math
import os
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


GAMMA = 0.99
ENTROPY_COEF = 0.01
LEARNING_RATE = 3.0e-3
MAX_GRAD_NORM = 1.0
NUM_ACTIONS = 9
DEFAULT_BASE_SEED = 18_473
CURRICULUM_GENERATIONS_PER_STAGE = 3

Rollout = Mapping[str, Any]
RunGeneration = Callable[..., Awaitable[Sequence[Rollout]] | Sequence[Rollout]]
GenerationCallback = Callable[
    [Mapping[str, Any], Sequence[Rollout]],
    Awaitable[None] | None,
]
LifecycleCallback = Callable[[Mapping[str, Any]], Awaitable[None] | None]
RolloutValidator = Callable[
    [int, Sequence[int], Sequence[Rollout]],
    Awaitable[None] | None,
]


class RolloutValidationError(ValueError):
    """Raised when a rollout cannot safely be used for an on-policy update."""


class PolicyUpdateError(RuntimeError):
    """Raised when an optimizer step cannot truthfully create a new policy."""


def compute_reward_to_go(
    rewards: Sequence[float] | Tensor,
    gamma: float = GAMMA,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return discounted reward-to-go for one episode.

    The calculation runs backwards and never crosses an episode boundary.  A
    one-dimensional empty reward sequence is accepted and produces an empty
    tensor; batch validation rejects empty episodes before an optimizer step.
    """

    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in the interval [0, 1]")

    try:
        reward_tensor = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=device,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("rewards must be a one-dimensional numeric sequence") from exc

    if reward_tensor.ndim != 1:
        raise ValueError("rewards must be a one-dimensional numeric sequence")
    if not torch.isfinite(reward_tensor).all().item():
        raise ValueError("rewards must contain only finite numbers")

    returns = torch.empty_like(reward_tensor)
    running_return = torch.zeros((), dtype=reward_tensor.dtype, device=reward_tensor.device)
    for index in range(reward_tensor.numel() - 1, -1, -1):
        running_return = reward_tensor[index] + gamma * running_return
        returns[index] = running_return
    return returns


def normalize_advantages(returns: Tensor, epsilon: float = 1.0e-8) -> Tensor:
    """Normalize a generation's returns when their variance is informative.

    A singleton remains unchanged.  A constant multi-step batch is centred to
    zero rather than amplified by division through a tiny standard deviation.
    """

    if not isinstance(returns, Tensor) or returns.ndim != 1:
        raise ValueError("returns must be a one-dimensional torch.Tensor")
    if returns.numel() == 0:
        raise ValueError("returns must not be empty")
    if not torch.isfinite(returns).all().item():
        raise ValueError("returns must contain only finite numbers")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("epsilon must be a positive finite number")
    if returns.numel() == 1:
        return returns.clone()

    centred = returns - returns.mean()
    standard_deviation = returns.std(unbiased=False)
    if standard_deviation.item() <= epsilon:
        return centred
    return centred / (standard_deviation + epsilon)


def create_optimizer(
    model: nn.Module,
    learning_rate: float = LEARNING_RATE,
) -> torch.optim.Adam:
    """Create the persistent Adam optimizer used across policy generations."""

    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a positive finite number")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model must have at least one trainable parameter")
    return torch.optim.Adam(parameters, lr=learning_rate)


def initialize_uniform_policy_v0(model: nn.Module) -> None:
    """Make the neural v0 distribution match the worker's null-weight policy.

    Daytona defines policy version zero as uniform seeded exploration with a
    null weight payload.  Zeroing only the final logits layer makes the local
    controller model exactly uniform too, so the first REINFORCE update remains
    on-policy while leaving the learned feature layers ready for later updates.
    """

    action_head = getattr(model, "action_head", None)
    if not isinstance(action_head, nn.Linear) or action_head.out_features != NUM_ACTIONS:
        raise ValueError("policy v0 requires a nine-output Linear action_head")
    with torch.no_grad():
        action_head.weight.zero_()
        if action_head.bias is not None:
            action_head.bias.zero_()


def reinforce_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollouts: Sequence[Rollout],
    *,
    gamma: float = GAMMA,
    entropy_coef: float = ENTROPY_COEF,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> dict[str, float | int | bool]:
    """Perform one robust batch REINFORCE update.

    All observations, categorical actions, and rewards are validated before
    model evaluation or parameter mutation.  Advantages are reward-to-go values
    normalized across all transitions in the generation.  The optimized loss
    is ``-E[A_t log pi(a_t|s_t)] - entropy_coef * E[H(pi(.|s_t))]``.
    """

    entropy_coef = float(entropy_coef)
    max_grad_norm = float(max_grad_norm)
    if not math.isfinite(entropy_coef) or entropy_coef < 0.0:
        raise ValueError("entropy_coef must be a finite non-negative number")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be a positive finite number")

    device = _model_device(model)
    observations, actions, returns, episode_count = _validated_batch(
        rollouts,
        gamma=gamma,
        device=device,
    )
    advantages = normalize_advantages(returns).detach()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    parameters_before = [
        parameter.detach().clone() for parameter in trainable_parameters
    ]

    optimizer.zero_grad(set_to_none=True)
    logits = model(observations)
    if logits.ndim != 2 or logits.shape != (observations.shape[0], NUM_ACTIONS):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError(
            "policy must return logits shaped "
            f"[{observations.shape[0]}, {NUM_ACTIONS}], got {list(logits.shape)}"
        )
    if not torch.isfinite(logits).all().item():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("policy produced non-finite logits")

    distribution = Categorical(logits=logits)
    selected_log_probabilities = distribution.log_prob(actions)
    entropy = distribution.entropy().mean()
    policy_loss = -(advantages * selected_log_probabilities).mean()
    loss = policy_loss - entropy_coef * entropy
    if not torch.isfinite(loss).item():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("REINFORCE loss is non-finite")

    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
        error_if_nonfinite=False,
    )
    if not torch.isfinite(torch.as_tensor(gradient_norm)).item():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("policy gradient is non-finite")
    optimizer.step()

    changed_parameter_tensors = 0
    changed_parameter_elements = 0
    squared_parameter_delta = 0.0
    max_parameter_delta = 0.0
    with torch.no_grad():
        for before, after in zip(parameters_before, trainable_parameters):
            if not torch.isfinite(after).all().item():
                raise FloatingPointError(
                    "optimizer produced non-finite policy parameters"
                )
            delta = after.detach() - before
            changed_elements = int(torch.count_nonzero(delta).item())
            if changed_elements:
                changed_parameter_tensors += 1
                changed_parameter_elements += changed_elements
                squared_parameter_delta += float(
                    torch.sum(delta.to(dtype=torch.float64) ** 2).item()
                )
                max_parameter_delta = max(
                    max_parameter_delta,
                    float(torch.max(torch.abs(delta)).item()),
                )

    if changed_parameter_elements == 0:
        raise PolicyUpdateError(
            "REINFORCE produced no parameter change; policy version was not advanced"
        )

    return {
        "weights_changed": True,
        "changed_parameter_tensors": changed_parameter_tensors,
        "changed_parameter_elements": changed_parameter_elements,
        "parameter_l2_delta": math.sqrt(squared_parameter_delta),
        "parameter_max_abs_delta": max_parameter_delta,
        "episodes": episode_count,
        "transitions": int(observations.shape[0]),
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "gradient_norm": float(torch.as_tensor(gradient_norm).detach().item()),
        "mean_reward_to_go": float(returns.mean().item()),
        "std_reward_to_go": float(returns.std(unbiased=False).item()),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()),
    }


def summarize_generation(rollouts: Sequence[Rollout]) -> dict[str, float | int | None]:
    """Calculate the judge-facing metrics for one Daytona generation."""

    if isinstance(rollouts, (str, bytes, Mapping)) or not isinstance(rollouts, Sequence):
        raise RolloutValidationError("rollouts must be a non-empty sequence of mappings")
    if len(rollouts) == 0:
        raise RolloutValidationError("rollouts must not be empty")

    totals: list[float] = []
    successes = 0
    collisions = 0
    clearances: list[float] = []
    for index, rollout in enumerate(rollouts):
        if not isinstance(rollout, Mapping):
            raise RolloutValidationError(f"rollout {index} must be a mapping")
        total = _episode_total_reward(rollout, index)
        totals.append(total)
        successes += int(bool(rollout.get("success", False)))
        collisions += int(_is_collision(rollout))

        clearance = _extract_min_clearance(rollout)
        if clearance is not None:
            clearances.append(clearance)

    count = len(rollouts)
    return {
        "worlds": count,
        "average_reward": sum(totals) / count,
        "best_reward": max(totals),
        "success_rate": successes / count,
        "collision_rate": collisions / count,
        "average_min_clearance": (
            sum(clearances) / len(clearances) if clearances else None
        ),
    }


def generation_seeds(base_seed: int, generation: int, worlds: int) -> list[int]:
    """Derive distinct seeds from the curriculum bands without global RNG state.

    Universe difficulty is encoded by the seed itself, so a recorded seed still
    reconstructs its exact world with no hidden generation parameter.  The
    first three generations use level 0; each three-generation stage admits
    one additional level until the complete procedural distribution is active.
    """

    base_seed = int(base_seed)
    generation = int(generation)
    worlds = int(worlds)
    if generation < 0:
        raise ValueError("generation must be non-negative")
    if worlds <= 0:
        raise ValueError("worlds must be positive")

    curriculum_level_count, curriculum_level_for_seed = _load_curriculum_contract()

    maximum_level = min(
        curriculum_level_count - 1,
        generation // CURRICULUM_GENERATIONS_PER_STAGE,
    )

    # Separate deterministic streams for each generation, then reject seed
    # bands above the current stage.  Sampling remains without replacement.
    stream_seed = base_seed + generation * 1_000_003
    generator = random.Random(stream_seed)
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < worlds:
        candidate = generator.randrange(0, 2**31)
        if candidate in seen or curriculum_level_for_seed(candidate) > maximum_level:
            continue
        seen.add(candidate)
        seeds.append(candidate)
    return seeds


async def run_daytona_training(
    *,
    generations: int,
    worlds: int,
    max_steps: int,
    obs_dim: int | None = None,
    base_seed: int = DEFAULT_BASE_SEED,
    gamma: float = GAMMA,
    entropy_coef: float = ENTROPY_COEF,
    learning_rate: float = LEARNING_RATE,
    max_grad_norm: float = MAX_GRAD_NORM,
    checkpoint_dir: str | Path = "checkpoints",
    start_generation: int = 0,
    initial_policy_version: int = 0,
    policy_checkpoint: str | Path | None = None,
    snapshot_name: str | None = None,
    keep_sandboxes: bool = False,
    event_callback: LifecycleCallback | None = None,
    rollout_validator: RolloutValidator | None = None,
    on_generation: GenerationCallback | None = None,
) -> list[dict[str, Any]]:
    """Train against real rollouts returned by the Daytona orchestrator only.

    ``on_generation`` is an integration hook, not a second training pathway.
    It receives the completed metrics/checkpoint record and the untouched real
    rollout structures after the policy update, allowing a controller to
    persist renderer-ready provenance without duplicating this loop.
    """

    generations = _positive_int(generations, "generations")
    worlds = _positive_int(worlds, "worlds")
    max_steps = _positive_int(max_steps, "max_steps")
    start_generation = _non_negative_int(start_generation, "start_generation")
    initial_policy_version = _non_negative_int(
        initial_policy_version,
        "initial_policy_version",
    )
    if start_generation != initial_policy_version:
        raise ValueError(
            "start_generation and initial_policy_version must match for the "
            "one-generation-per-policy demo contract"
        )
    if start_generation == 0 and policy_checkpoint is not None:
        raise ValueError("a fresh v0 run must not supply policy_checkpoint")
    if start_generation > 0 and policy_checkpoint is None:
        raise ValueError("a resumed vN run requires policy_checkpoint")
    observation_dimension = _resolve_observation_dim(obs_dim)

    # Both imports stay inside the real async training entry point.  Importing
    # trainer utilities for tests therefore neither requires Daytona nor starts
    # any rollout infrastructure.
    run_generation = _load_daytona_run_generation()
    try:
        from rl_policy import create_policy, encode_policy_weights
    except ImportError as exc:
        raise RuntimeError(
            "rl_policy is required for Daytona training; install project dependencies"
        ) from exc

    checkpoint_root = Path(checkpoint_dir)
    model = create_policy(observation_dimension, seed=int(base_seed))
    optimizer = create_optimizer(model, learning_rate)
    if start_generation == 0:
        initialize_uniform_policy_v0(model)
        input_checkpoint_path = checkpoint_root / "policy_v000.pt"
        _save_checkpoint(
            input_checkpoint_path,
            model=model,
            optimizer=optimizer,
            policy_version=0,
            obs_dim=observation_dimension,
        )
    else:
        input_checkpoint_path = Path(policy_checkpoint)
        _load_checkpoint(
            input_checkpoint_path,
            model=model,
            optimizer=optimizer,
            expected_policy_version=initial_policy_version,
            expected_obs_dim=observation_dimension,
        )

    history: list[dict[str, Any]] = []
    for generation in range(start_generation, start_generation + generations):
        policy_version = initial_policy_version + (generation - start_generation)
        seeds = generation_seeds(base_seed, generation, worlds)
        # Frozen Daytona contract: v0 is seeded uniform exploration and must
        # carry null weights.  After its on-policy update, every learned
        # version is encoded with the safe JSON/base64 transport.
        policy_weights = (
            None if policy_version == 0 else encode_policy_weights(model)
        )
        run_kwargs: dict[str, Any] = {
            "policy_version": policy_version,
            "policy_weights": policy_weights,
            "seeds": seeds,
            "max_steps": max_steps,
            "keep_sandboxes": bool(keep_sandboxes),
        }
        if snapshot_name is not None:
            run_kwargs["snapshot_name"] = snapshot_name
        if event_callback is not None:
            def tagged_event(
                event: Mapping[str, Any],
                *,
                generation: int = generation,
                frozen_policy_version: int = policy_version,
            ) -> Awaitable[None] | None:
                tagged = dict(event)
                tagged["generation"] = generation
                tagged["policy_version"] = frozen_policy_version
                return event_callback(tagged)

            run_kwargs["event_callback"] = tagged_event

        generation_result = run_generation(
            **run_kwargs,
        )
        if inspect.isawaitable(generation_result):
            generation_result = await generation_result
        rollouts = _coerce_generation_result(generation_result)
        if rollout_validator is not None:
            validation_result = rollout_validator(
                policy_version,
                tuple(seeds),
                tuple(rollouts),
            )
            if inspect.isawaitable(validation_result):
                await validation_result

        generation_metrics = summarize_generation(rollouts)
        update_metrics = reinforce_update(
            model,
            optimizer,
            rollouts,
            gamma=gamma,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )
        next_version = policy_version + 1
        _, curriculum_level_for_seed = _load_curriculum_contract()
        curriculum_levels = [curriculum_level_for_seed(seed) for seed in seeds]
        checkpoint_path = checkpoint_root / f"policy_v{next_version:03d}.pt"
        _save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            policy_version=next_version,
            obs_dim=observation_dimension,
        )

        record = {
            "generation": generation,
            "policy_version": policy_version,
            "next_policy_version": next_version,
            "seeds": seeds,
            "curriculum_levels": curriculum_levels,
            "curriculum_max_level": max(curriculum_levels),
            "input_checkpoint": str(input_checkpoint_path),
            "checkpoint": str(checkpoint_path),
            **generation_metrics,
            **update_metrics,
        }
        history.append(record)
        if on_generation is not None:
            callback_result = on_generation(dict(record), tuple(rollouts))
            if inspect.isawaitable(callback_result):
                await callback_result
        _print_generation(record)
        input_checkpoint_path = checkpoint_path

    return history


def _validated_batch(
    rollouts: Sequence[Rollout],
    *,
    gamma: float,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, int]:
    if isinstance(rollouts, (str, bytes, Mapping)) or not isinstance(rollouts, Sequence):
        raise RolloutValidationError("rollouts must be a non-empty sequence of mappings")
    if len(rollouts) == 0:
        raise RolloutValidationError("rollouts must not be empty")

    observation_batches: list[Tensor] = []
    action_batches: list[Tensor] = []
    return_batches: list[Tensor] = []
    expected_obs_dim: int | None = None

    for episode_index, rollout in enumerate(rollouts):
        if not isinstance(rollout, Mapping):
            raise RolloutValidationError(f"rollout {episode_index} must be a mapping")
        missing = [key for key in ("observations", "actions", "rewards") if key not in rollout]
        if missing:
            raise RolloutValidationError(
                f"rollout {episode_index} is missing required field(s): {', '.join(missing)}"
            )

        observations = _as_tensor(
            rollout["observations"],
            dtype=torch.float32,
            device=device,
            label=f"rollout {episode_index} observations",
        )
        if observations.ndim != 2:
            raise RolloutValidationError(
                f"rollout {episode_index} observations must have shape [steps, obs_dim]"
            )
        if observations.shape[0] == 0 or observations.shape[1] == 0:
            raise RolloutValidationError(f"rollout {episode_index} must not be empty")
        if not torch.isfinite(observations).all().item():
            raise RolloutValidationError(
                f"rollout {episode_index} observations contain non-finite values"
            )
        if expected_obs_dim is None:
            expected_obs_dim = int(observations.shape[1])
        elif observations.shape[1] != expected_obs_dim:
            raise RolloutValidationError(
                f"rollout {episode_index} observation dimension {observations.shape[1]} "
                f"does not match expected dimension {expected_obs_dim}"
            )

        raw_actions = _as_tensor(
            rollout["actions"],
            dtype=torch.float64,
            device=device,
            label=f"rollout {episode_index} actions",
        )
        if raw_actions.ndim != 1:
            raise RolloutValidationError(
                f"rollout {episode_index} actions must be one-dimensional categorical indices"
            )
        if not torch.isfinite(raw_actions).all().item():
            raise RolloutValidationError(
                f"rollout {episode_index} actions contain non-finite values"
            )
        if not torch.equal(raw_actions, raw_actions.trunc()):
            raise RolloutValidationError(
                f"rollout {episode_index} actions must be integer categorical indices"
            )
        actions = raw_actions.to(dtype=torch.long)
        if ((actions < 0) | (actions >= NUM_ACTIONS)).any().item():
            raise RolloutValidationError(
                f"rollout {episode_index} actions must be in [0, {NUM_ACTIONS - 1}]"
            )

        rewards = _as_tensor(
            rollout["rewards"],
            dtype=torch.float32,
            device=device,
            label=f"rollout {episode_index} rewards",
        )
        if rewards.ndim != 1:
            raise RolloutValidationError(
                f"rollout {episode_index} rewards must be one-dimensional"
            )
        if not torch.isfinite(rewards).all().item():
            raise RolloutValidationError(
                f"rollout {episode_index} rewards contain non-finite values"
            )

        step_count = observations.shape[0]
        if actions.shape[0] != step_count or rewards.shape[0] != step_count:
            raise RolloutValidationError(
                f"rollout {episode_index} lengths differ: "
                f"observations={step_count}, actions={actions.shape[0]}, "
                f"rewards={rewards.shape[0]}"
            )

        observation_batches.append(observations)
        action_batches.append(actions)
        return_batches.append(compute_reward_to_go(rewards, gamma, device=device))

    return (
        torch.cat(observation_batches, dim=0),
        torch.cat(action_batches, dim=0),
        torch.cat(return_batches, dim=0),
        len(rollouts),
    )


def _as_tensor(
    value: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
    label: str,
) -> Tensor:
    try:
        return torch.as_tensor(value, dtype=dtype, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RolloutValidationError(f"{label} must be a rectangular numeric sequence") from exc


def _model_device(model: nn.Module) -> torch.device:
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("model must have at least one trainable parameter")
    return trainable[0].device


def _episode_total_reward(rollout: Rollout, index: int) -> float:
    if "reward" in rollout:
        try:
            total = float(rollout["reward"])
        except (TypeError, ValueError) as exc:
            raise RolloutValidationError(f"rollout {index} reward must be numeric") from exc
    elif "rewards" in rollout:
        try:
            total = sum(float(reward) for reward in rollout["rewards"])
        except (TypeError, ValueError) as exc:
            raise RolloutValidationError(
                f"rollout {index} rewards must be a numeric sequence"
            ) from exc
    else:
        raise RolloutValidationError(
            f"rollout {index} must contain reward or rewards for metrics"
        )
    if not math.isfinite(total):
        raise RolloutValidationError(f"rollout {index} total reward must be finite")
    return total


def _is_collision(rollout: Rollout) -> bool:
    if bool(rollout.get("collision", False)):
        return True
    termination = rollout.get("termination", rollout.get("status", ""))
    if isinstance(termination, str) and "collision" in termination.lower():
        return True
    info = rollout.get("info")
    return isinstance(info, Mapping) and bool(info.get("collision"))


def _extract_min_clearance(rollout: Rollout) -> float | None:
    value = rollout.get("min_clearance")
    if value is None:
        info = rollout.get("info")
        if isinstance(info, Mapping):
            value = info.get("min_clearance")
    if value is None:
        return None
    try:
        clearance = float(value)
    except (TypeError, ValueError) as exc:
        raise RolloutValidationError("min_clearance must be numeric when provided") from exc
    if not math.isfinite(clearance):
        raise RolloutValidationError("min_clearance must be finite when provided")
    return clearance


def _resolve_observation_dim(obs_dim: int | None) -> int:
    if obs_dim is None:
        try:
            from gravity_env import OBSERVATION_DIM
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "gravity_env.OBSERVATION_DIM is required; alternatively pass --obs-dim"
            ) from exc
        obs_dim = OBSERVATION_DIM
    return _positive_int(obs_dim, "obs_dim")


def _load_curriculum_contract() -> tuple[int, Callable[[int], int]]:
    """Load headless seed metadata without constructing a local environment."""

    try:
        from gravity_env import CURRICULUM_LEVELS, curriculum_level_for_seed
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("gravity curriculum metadata is unavailable") from exc
    return CURRICULUM_LEVELS, curriculum_level_for_seed


def _load_daytona_run_generation() -> RunGeneration:
    try:
        from daytona_orchestrator import run_generation
    except ImportError as exc:
        raise RuntimeError(
            "Daytona training requires daytona_orchestrator.run_generation; "
            "there is intentionally no local rollout fallback"
        ) from exc
    if not callable(run_generation):
        raise RuntimeError("daytona_orchestrator.run_generation must be callable")
    return run_generation


def _coerce_generation_result(value: Any) -> Sequence[Rollout]:
    if isinstance(value, Mapping) and "rollouts" in value:
        value = value["rollouts"]
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise RolloutValidationError(
            "daytona_orchestrator.run_generation must return a rollout sequence"
        )
    if not value:
        raise RolloutValidationError("Daytona returned an empty generation")
    return value


def _load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_policy_version: int,
    expected_obs_dim: int,
) -> None:
    """Load a complete model-and-optimizer continuation checkpoint."""

    if not path.is_file():
        raise ValueError(f"policy checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"could not load policy checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"policy checkpoint {path} must contain a mapping")
    version = payload.get("policy_version")
    if isinstance(version, bool) or version != int(expected_policy_version):
        raise ValueError(
            f"policy checkpoint {path} has version {version!r}; "
            f"expected {expected_policy_version}"
        )
    obs_dim = payload.get("obs_dim")
    if isinstance(obs_dim, bool) or obs_dim != int(expected_obs_dim):
        raise ValueError(
            f"policy checkpoint {path} has obs_dim {obs_dim!r}; "
            f"expected {expected_obs_dim}"
        )
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError(f"policy checkpoint {path} has no model_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError(f"policy checkpoint {path} has no optimizer_state_dict")
    saved_optimizer_state = optimizer_state.get("state")
    if not isinstance(saved_optimizer_state, Mapping) or not saved_optimizer_state:
        raise ValueError(
            f"policy checkpoint {path} has no learned optimizer state; "
            "refusing to reset Adam during resume"
        )
    try:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"policy checkpoint {path} is incompatible: {exc}") from exc
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise ValueError(f"policy checkpoint {path} has non-finite model weights")
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                raise ValueError(
                    f"policy checkpoint {path} has non-finite optimizer state"
                )


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    policy_version: int,
    obs_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_version": int(policy_version),
        "obs_dim": int(obs_dim),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _print_generation(record: Mapping[str, Any]) -> None:
    average_clearance = record["average_min_clearance"]
    clearance_text = "n/a" if average_clearance is None else f"{average_clearance:.3f}"
    print()
    print(f"GENERATION {record['generation']:02d}")
    print()
    print(f"Policy: v{record['policy_version']}")
    print(f"Daytona worlds: {record['worlds']}")
    print(f"Average reward: {record['average_reward']:.3f}")
    print(f"Best reward: {record['best_reward']:.3f}")
    print(f"Success rate: {record['success_rate']:.1%}")
    print(f"Collision rate: {record['collision_rate']:.1%}")
    print(f"Average min clearance: {clearance_text}")
    print(f"Policy updated → v{record['next_policy_version']}")


def _positive_int(value: Any, name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _non_negative_int(value: Any, name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Gravity Gauntlet from real Daytona rollout generations."
    )
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--obs-dim", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--start-generation", type=int, default=0)
    parser.add_argument("--initial-policy-version", type=int, default=0)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--snapshot-name")
    parser.add_argument("--keep-sandboxes", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        asyncio.run(
            run_daytona_training(
                generations=args.generations,
                worlds=args.worlds,
                max_steps=args.max_steps,
                obs_dim=args.obs_dim,
                base_seed=args.base_seed,
                gamma=args.gamma,
                entropy_coef=args.entropy_coef,
                learning_rate=args.learning_rate,
                max_grad_norm=args.max_grad_norm,
                checkpoint_dir=args.checkpoint_dir,
                start_generation=args.start_generation,
                initial_policy_version=args.initial_policy_version,
                policy_checkpoint=args.policy_checkpoint,
                snapshot_name=args.snapshot_name,
                keep_sandboxes=args.keep_sandboxes,
            )
        )
    except (FloatingPointError, RuntimeError, RolloutValidationError, ValueError) as exc:
        raise SystemExit(f"training failed: {exc}") from exc


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_BASE_SEED",
    "ENTROPY_COEF",
    "GAMMA",
    "GenerationCallback",
    "LEARNING_RATE",
    "MAX_GRAD_NORM",
    "PolicyUpdateError",
    "LifecycleCallback",
    "RolloutValidationError",
    "RolloutValidator",
    "compute_reward_to_go",
    "create_optimizer",
    "generation_seeds",
    "initialize_uniform_policy_v0",
    "normalize_advantages",
    "reinforce_update",
    "run_daytona_training",
    "summarize_generation",
]
