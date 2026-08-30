# Gravity Gauntlet judge-facing UI specification

## Purpose

`visual_demo.py` is a provenance-safe visual replay for the Gravity Gauntlet
training pipeline. It tells one evidence-backed story:

```text
one frozen policy
        -> many distinct Daytona sandboxes and seeded universes
        -> real recorded trajectories, actions, outcomes, and rewards
        -> one verified policy update
        -> the next policy version
```

The renderer does not simulate remote work, alter recorded trajectories, run
the optimizer, or invent learning progress. `GravityEnv` remains the authority
for universe geometry and local physics. The controller generation JSON remains
the authority for Daytona identity, lifecycle, trajectories, rewards, metrics,
champion selection, and policy-update evidence.

## Run modes

The command line requires exactly one explicit source mode.

### Recorded replay

```bash
python3 visual_demo.py \
  --rollouts runs/generation_000.json \
  --generation 0 \
  --policy-version 0
```

The champion is selected by default. `--seed` may select another universe in
the same evaluated generation and policy version. The renderer never mixes
world cards or summary metrics from different generation/policy groups.

`runs/training_state.json` is also accepted. Its retained generations feed the
real learning-history chart, while the latest generation/policy is selected for
the main replay unless the operator selects another one.

### Local development preview

```bash
python3 visual_demo.py --local-preview --seed 7
```

This is keyboard-driven local `GravityEnv` physics. It is labelled `LOCAL
PREVIEW` in the title bar, main banner, provenance panel, and metrics panel. It
never displays a generation, trained-policy transition, Daytona lifecycle, or
sandbox identity. `--generation` and `--policy-version` are rejected in this
mode.

There is intentionally no fake “train next generation” button. Real training
is an external controller operation documented in `JUDGE_DEMO.md`; the replay
only visualizes the artifact produced after that operation succeeds.

## Verified Daytona contract

A controller generation is labelled `DAYTONA TRAINING` only when all of these
checks pass:

- The generation status is `COMPLETE`, `world_count` matches the non-empty
  `worlds` collection, and all worlds have distinct indices, seeds, and
  non-synthetic sandbox IDs.
- Generation is a non-negative canonical policy generation, `policy_version`
  equals `generation`, `policy_version_used` repeats that evaluated version,
  and world indices are exactly one through the declared world count.
- Every world declares the `daytona` execution backend and the evaluated policy
  version matches the generation policy.
- Every world has a finite reward, a consistent success/termination pair, at
  least two trajectory samples, at least one categorical action in the valid
  range, the same number of finite decoded action vectors, a matching terminal
  status, and a recorded universe.
- Each lifecycle is exactly `CREATING -> LIVE -> RUNNING -> terminal outcome ->
  RESULT_COLLECTED`. Every event must identify the same generation, policy,
  world index, and seed. `CREATING` has no sandbox ID; every subsequent event
  must carry that world's exact sandbox ID.
- The seed batch, average reward, best reward, success rate, and collision rate
  recompute exactly from the displayed worlds.
- The declared champion points to the maximum recorded reward and matches its
  generation, world index, seed, sandbox ID, policy version, reward, outcome,
  actions, execution backend, and trajectory.
- The next policy version is exactly the evaluated version plus one.
- The training envelope names a checkpoint and contains real training fields.
- `extra.policy_update` proves `weights_changed: true`, names both checkpoints,
  binds them to `policy_vNNN.pt` and the next version, binds the trainer output
  to that next checkpoint, and carries two distinct lowercase 64-character
  model-state SHA-256 digests.

The direct `daytona_orchestrator` result envelope can be labelled `DAYTONA
ROLLOUT` after its own IDs, seeds, policy, universe, trajectories, valid actions,
summary metrics, ordered seed/ID lists, trajectory count, concurrency flag,
cleanup result, wall-clock value, and best-sandbox checks pass. It cannot claim
a trained next policy because that envelope contains no trainer update proof.

An artifact with sandbox-like metadata that does not pass the full applicable
gate remains visible only as `UNVERIFIED RECORDED REPLAY`. Missing or malformed
trajectory coordinates fail loading rather than becoming a fabricated path.

## Universe and trajectory integrity

Before rendering any loaded world or historical ghost, the UI reconstructs
`GravityEnv(seed=<recorded seed>)` and compares its complete `universe_dict()`
with the universe recorded by the worker. A mismatch stops replay with a clear
stale-snapshot error. It is never patched, silently accepted, or replaced with
another seed. Every artifact supplied through `--rollouts` must contain its
recorded universe. The interactive `--local-preview` mode creates its own
current local environment directly and never routes local motion through the
recorded-replay helper.

Trajectory points, velocity, clearance, reward samples, and action vectors are
read from the artifact. The display applies only a uniform screen transform to
fit the original 1200 by 800 world into the unobscured 750 by 500 hero viewport;
stored values and physics coordinates are unchanged.

The active path is revealed progressively over approximately eight seconds,
then its terminal outcome is held before cycling. The ship angle, motion streaks,
and exhaust are derived from recorded velocity and policy action vectors. The
full future of the active path is never drawn as a ghost.

Ghost trajectories obey both rules below:

1. The seed must exactly match the active universe.
2. The generation must precede the active generation when both are known.

This prevents a path from one geometry being projected over another and
prevents future/active trajectory leakage.

## Screen anatomy

The 1200 by 800 composition has five fixed regions.

### Replay identity — upper left

The banner names Gravity Gauntlet and makes the current state unmistakable:

- `CURRENT CHAMPION` for the verified maximum-reward Daytona world;
- `PARALLEL WORLD N` for another verified world;
- `LOCAL PREVIEW` for local-only data; or
- `UNVERIFIED REPLAY` when provenance did not pass a remote proof gate.

It also reports the real world index, generation, evaluated policy version,
terminal outcome, and reward when those values exist.

### Execution provenance — upper centre

For verified controller data, this is a five-stage recorded lifecycle strip.
It is explicitly historical evidence, not a simulated live-status animation.
For local preview it reads `LOCAL PREVIEW — NO DAYTONA CLAIM`; for incomplete
recorded data it says that no lifecycle events were present.

### Hero universe — centre left

The largest area shows the active seeded universe and recorded flight:

- glowing gravity influence rings and shaded planets;
- deterministic asteroid geometry;
- pulsing target portal;
- layered, fading trajectory trail;
- velocity-based motion streaks;
- policy-vector exhaust and velocity-oriented ship; and
- high-contrast success, collision, timeout, or out-of-bounds terminal effects.

The terminal overlay remains inside this viewport and does not cover metrics or
parallel universe cards.

### Parallel universes — right rail

Up to eight seed-correct universe cards are visible simultaneously. Each card
contains its own reconstructed planets, asteroids, portal, and complete path,
plus:

- real world index;
- recorded reward and outcome;
- execution provenance;
- compact sandbox ID preserving the beginning and end; and
- champion, active, preview-best, or recorded-best status as appropriate.

The selected world's full sandbox ID is also shown in the lower metrics panel
when it fits. An unusually long ID is shortened only by preserving both ends,
so the compact card label never silently replaces judge-verifiable identity.

### Metrics, history, and controls — lower band

The replay metrics panel reports the selected generation/policy group only:

- world count and execution class;
- generation average and best reward;
- success and collision rates;
- reward accumulated so far and final reward;
- recorded speed and current/minimum clearance;
- world index, seed, trajectory cursor, terminal state; and
- full sandbox identity when it fits, otherwise a clearly endpoint-preserving
  shortened identity.

`TRAINED vN` appears only beside a fully verified controller policy transition.
It means the displayed experiences were used to update the evaluated policy
into that next version; it does not claim that the next version has already
been evaluated.

With two or more loaded generations, the adjacent chart plots actual average
and best reward. It preserves regressions and flat results rather than forcing
an upward line. Its heading states whether the data is verified Daytona,
explicit local preview, or unverified recorded history.

Controls are:

- `R`: restart the current recorded replay, or restart the local world;
- `N`: select the next recorded universe, or create the next local seed;
- `WASD` / arrow keys: local-preview thrust only; and
- `Esc`: exit.

## Champion semantics

There is exactly one champion per generation/policy group. A declared champion
must have the maximum recorded reward or loading fails. If a compact/local
artifact does not declare one, the UI marks the maximum finite recorded reward
as the group best without upgrading its provenance.

## Performance and headless QA

Rendering targets 60 frames per second. Backgrounds, stars, and mini-universe
geometry are deterministic for a seed; display animation does not mutate the
environment or artifact. `GRAVITY_DEMO_MAX_FRAMES=N` bounds either mode for a
headless smoke test, for example:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
GRAVITY_DEMO_MAX_FRAMES=3 \
python3 visual_demo.py --local-preview --seed 7
```

UI contract tests live in `tests/test_visual_demo.py`. They cover successful
controller loading, exact IDs/metrics/lifecycle retention, champion selection,
malformed lifecycle and trajectory rejection, policy-update proof, multi-
generation history, seed-safe ghosts, explicit CLI boundaries, and the absence
of Pygame from worker/runtime requirements.

## Failure behaviour

The UI fails closed for malformed JSON, non-finite coordinates or rewards,
empty trajectories, impossible rate values, inconsistent champion declarations,
missing selected generation/policy/seed, and stale recorded universe geometry.
It reports the reason through the command-line parser and exits; it does not
fall back from a requested replay to local simulation.
