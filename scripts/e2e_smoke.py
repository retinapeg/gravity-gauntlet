#!/usr/bin/env python3
"""Real-only end-to-end smoke test for Gravity Gauntlet.

This script imports the production controller and therefore reaches Daytona's
real ``run_generation`` implementation.  It contains no mock runner and no
local physics fallback.  The default two-world run is intentionally small;
pass ``--worlds 8`` for the full judge-demo fan-out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from gravity_gauntlet.demo_controller import (
    DEFAULT_BASE_SEED,
    IntegrationContractError,
    IntegrationUnavailableError,
    checkpoint_model_digest,
    run_training_demo,
)


class SmokeFailure(RuntimeError):
    """Raised when the real E2E path finishes without the required proof."""


def _validate_world(world: Mapping[str, Any], *, expected_policy: int) -> None:
    world_index = world.get("world_index", "?")
    sandbox_id = world.get("sandbox_id")
    if not isinstance(sandbox_id, str) or not sandbox_id.strip():
        raise SmokeFailure(f"world {world_index} has no real Daytona sandbox ID")
    if world.get("policy_version") != expected_policy:
        raise SmokeFailure(f"world {world_index} has the wrong policy version")

    reward = world.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise SmokeFailure(f"world {world_index} has no numeric reward")
    if not math.isfinite(float(reward)):
        raise SmokeFailure(f"world {world_index} reward is not finite")

    trajectory = world.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise SmokeFailure(f"world {world_index} has no trajectory")

    lifecycle = world.get("lifecycle")
    if not isinstance(lifecycle, list):
        raise SmokeFailure(f"world {world_index} has no lifecycle history")
    states = {
        event.get("state")
        for event in lifecycle
        if isinstance(event, Mapping)
    }
    if not {"LIVE", "RUNNING", "RESULT_COLLECTED"}.issubset(states):
        raise SmokeFailure(
            f"world {world_index} lifecycle does not prove a collected Daytona run"
        )

    extra = world.get("extra")
    if not isinstance(extra, Mapping):
        raise SmokeFailure(f"world {world_index} has no trainer rollout fields")
    observations = extra.get("observations")
    actions = world.get("actions")
    rewards = extra.get("rewards")
    if not all(isinstance(value, list) for value in (observations, actions, rewards)):
        raise SmokeFailure(
            f"world {world_index} is missing observations/actions/rewards"
        )
    if not observations or len(observations) != len(actions) or len(actions) != len(rewards):
        raise SmokeFailure(
            f"world {world_index} trainer rollout lengths are empty or inconsistent"
        )
    if any(isinstance(action, bool) or not isinstance(action, int) for action in actions):
        raise SmokeFailure(
            f"world {world_index} actions are not categorical integer indices"
        )


async def run_smoke(args: argparse.Namespace) -> Path:
    try:
        os.environ["DAYTONA_API_KEY"]
    except KeyError as exc:
        raise SmokeFailure(
            "DAYTONA_API_KEY is required; this smoke test will not substitute "
            "local rollouts"
        ) from exc

    training = await run_training_demo(
        generations=1,
        worlds=args.worlds,
        max_steps=args.max_steps,
        base_seed=args.base_seed,
        runs_dir=args.runs_dir,
        checkpoint_dir=args.checkpoint_dir,
        snapshot_name=args.snapshot_name,
        keep_sandboxes=args.keep_sandboxes,
        echo_lifecycle=True,
        allow_overwrite=args.overwrite,
    )
    if len(training.recent_generations) != 1:
        raise SmokeFailure("controller did not return exactly one generation")
    generation = training.recent_generations[0]
    document = generation.to_dict()
    worlds = document.get("worlds")
    if not isinstance(worlds, list) or len(worlds) != args.worlds:
        raise SmokeFailure(
            f"expected {args.worlds} Daytona worlds, received "
            f"{len(worlds) if isinstance(worlds, list) else 'invalid output'}"
        )
    seeds = [world.get("seed") for world in worlds if isinstance(world, Mapping)]
    if len(seeds) != args.worlds or len(set(seeds)) != args.worlds:
        raise SmokeFailure("real worlds did not use distinct universe seeds")
    sandbox_ids = [
        world.get("sandbox_id") for world in worlds if isinstance(world, Mapping)
    ]
    if len(sandbox_ids) != args.worlds or len(set(sandbox_ids)) != args.worlds:
        raise SmokeFailure("worlds did not come from distinct Daytona sandbox IDs")
    for world in worlds:
        if not isinstance(world, Mapping):
            raise SmokeFailure("controller returned a non-object world")
        _validate_world(world, expected_policy=generation.policy_version)

    if generation.next_policy_version != generation.policy_version + 1:
        raise SmokeFailure("trainer did not advance the policy version")
    training_metrics = generation.extra.get("training")
    if not isinstance(training_metrics, Mapping) or not training_metrics:
        raise SmokeFailure("trainer returned no policy-update metrics")
    if generation.extra.get("execution_backend") != "daytona":
        raise SmokeFailure("generation does not carry explicit Daytona provenance")

    initial_checkpoint_path = (
        Path(args.checkpoint_dir) / f"policy_v{generation.policy_version:03d}.pt"
    )
    next_checkpoint_path = (
        Path(args.checkpoint_dir)
        / f"policy_v{generation.next_policy_version:03d}.pt"
    )
    if not initial_checkpoint_path.is_file() or not next_checkpoint_path.is_file():
        raise SmokeFailure("trainer did not save the v0 and updated policy checkpoints")
    if generation.extra.get("trainer_checkpoint") != str(next_checkpoint_path):
        raise SmokeFailure("generation metadata does not identify its trainer checkpoint")
    initial_digest = checkpoint_model_digest(
        initial_checkpoint_path,
        expected_policy_version=generation.policy_version,
    )
    next_digest = checkpoint_model_digest(
        next_checkpoint_path,
        expected_policy_version=generation.next_policy_version,
    )
    if initial_digest == next_digest:
        raise SmokeFailure("trainer incremented the version without changing model weights")
    update_proof = generation.extra.get("policy_update")
    if not isinstance(update_proof, Mapping):
        raise SmokeFailure("generation has no model-weight update proof")
    if (
        update_proof.get("weights_changed") is not True
        or update_proof.get("input_model_sha256") != initial_digest
        or update_proof.get("next_model_sha256") != next_digest
    ):
        raise SmokeFailure("generation model-weight proof does not match its checkpoints")

    output_path = Path(args.runs_dir) / "generation_000.json"
    if not output_path.is_file():
        raise SmokeFailure(f"generation JSON was not saved at {output_path}")
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    if saved.get("world_count") != args.worlds:
        raise SmokeFailure("saved generation JSON has the wrong world count")
    if not saved.get("champion", {}).get("sandbox_id"):
        raise SmokeFailure("saved generation JSON has no real champion sandbox ID")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from visual_demo import load_rollout_trails

    attempts = load_rollout_trails(output_path)
    if len(attempts) != args.worlds:
        raise SmokeFailure("visual loader did not receive every Daytona world")
    if not all(attempt.daytona_verified for attempt in attempts):
        raise SmokeFailure("visual loader did not verify Daytona provenance")
    champions = [attempt for attempt in attempts if attempt.is_champion]
    if (
        len(champions) != 1
        or champions[0].sandbox_id
        != saved["champion"]["sandbox_id"]
    ):
        raise SmokeFailure("visual loader did not preserve the real champion")
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real Daytona -> trajectories -> REINFORCE smoke test."
    )
    parser.add_argument(
        "--worlds",
        type=int,
        default=2,
        help="2 for quick smoke; 8 for full judge fan-out",
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--snapshot-name", default="gravity-gauntlet-worker-v2")
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs/e2e_smoke")
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/e2e_smoke")
    )
    parser.add_argument("--keep-sandboxes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.worlds <= 0 or args.max_steps <= 0:
        raise SystemExit("E2E SMOKE FAILED: --worlds and --max-steps must be positive")
    try:
        output_path = asyncio.run(run_smoke(args))
    except (
        IntegrationContractError,
        IntegrationUnavailableError,
        SmokeFailure,
        ValueError,
    ) as exc:
        raise SystemExit(f"E2E SMOKE FAILED: {exc}") from exc

    print("\nE2E SMOKE PASSED")
    print(f"Real Daytona worlds: {args.worlds}")
    print(f"Saved generation: {output_path}")


if __name__ == "__main__":
    main()
