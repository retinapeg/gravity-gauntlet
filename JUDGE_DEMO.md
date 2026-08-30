# Gravity Gauntlet — judge demo

Run every command from the one canonical checkout:

```bash
cd /Users/leonardaarons-ditson/Desktop/gravity-gauntlet
```

Real execution requires `DAYTONA_API_KEY` in the shell environment. Never put
the key in this repository, a command argument, or a job JSON file.

## What to run

Use a fresh versioned snapshot name so the snapshot contains the current
physics, rollout, worker-bridge, and policy files:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install daytona==0.207.0
python3 daytona_snapshot.py \
  --snapshot-name gravity-gauntlet-worker-v3
```

Run one complete training generation—one frozen policy across eight real,
concurrent Daytona universes, followed by one REINFORCE update:

```bash
PYTHONPATH=src:. python3 -m gravity_gauntlet.demo_controller \
  --worlds 8 \
  --generations 1 \
  --max-steps 500 \
  --base-seed 18473 \
  --snapshot-name gravity-gauntlet-worker-v3
```

Load the saved real trajectories and replay the highest-reward rollout first,
with the other evaluated universes shown as parallel seed-correct cards. Read
the champion's real seed from the artifact to select it explicitly:

```bash
CHAMPION_SEED=$(python3 -c 'import json; print(json.load(open("runs/generation_000.json"))["champion"]["seed"])')
python3 visual_demo.py \
  --rollouts runs/generation_000.json \
  --generation 0 \
  --policy-version 0 \
  --seed "$CHAMPION_SEED"
```

The canonical trainer starts at **v0**, whose worker payload is exactly null
and whose action distribution is seeded uniform exploration. The first
successful REINFORCE update advances v0 to v1; v1 and later carry encoded
neural-policy weights. The controller only persists the trainer's completed
generation and does not own a second policy/version loop.

## Demo flow

One controller invocation performs this complete generation flow:

```text
one v0-null or v1+-encoded policy frozen by trainer.py
              |
              v
eight real Daytona sandboxes created concurrently
              |
              v
eight distinct deterministic universe seeds
              |
              v
canonical GravityEnv physics inside every sandbox
              |
              v
real trajectories + observations + actions + rewards collected
              |
              v
one batch REINFORCE update in trainer.py
              |
              v
next policy version + runs/generation_000.json
              |
              v
visual_demo.py replays the champion and parallel worlds over verified geometry
```

The terminal shows every world's real lifecycle:

```text
CREATING -> LIVE -> RUNNING -> terminal outcome -> RESULT_COLLECTED
```

The final panel reports average, best, and worst reward; success and collision
rates; the champion's real sandbox ID; and the policy version transition. If
Daytona or any world fails, the generation fails visibly. Partial results are
not trained, and local physics is never substituted.

A one-generation run displays the champion among the real v0 experiences used
to create v1; v1 has not been evaluated yet. To collect real progression data
across evaluated policies, run multiple generations in one process so the
same optimizer is preserved and use isolated output directories:

```bash
PYTHONPATH=src:. python3 -m gravity_gauntlet.demo_controller \
  --worlds 8 \
  --generations 4 \
  --max-steps 500 \
  --base-seed 18473 \
  --snapshot-name gravity-gauntlet-worker-v3 \
  --runs-dir runs/progression \
  --checkpoint-dir checkpoints/progression
```

`runs/progression/training_state.json` then contains every retained trajectory,
its generation, its evaluated policy version, and each generation's champion.
The renderer selects one evaluated generation/policy at a time, replays its
champion by default, and shows up to eight parallel worlds using each world's
own seed. Before drawing, it compares every recorded worker `universe` with the
current `GravityEnv(seed)` and refuses a stale-snapshot geometry mismatch. It
does not mix metrics or trajectories across generations in one panel.

Checkpoint resume is not implemented yet. Do not rerun a one-generation
command and describe it as continuing v1: the controller deliberately refuses
existing targets, and `--overwrite` starts again from v0. Use one uninterrupted
`--generations N` invocation for genuine v0 → v1 → ... progression.

## 30-second pitch

“Traditional reinforcement learning lets an agent experience one environment
at a time. Gravity Gauntlet uses Daytona to create a fleet of isolated
gravitational universes, allowing the same policy to gather diverse physical
experiences simultaneously. We combine those experiences to improve the
policy and launch the next generation.”

## Emergency commands

### One-world Daytona gate

This verifies one real sandbox using the v0 seeded baseline before training:

```bash
python3 daytona_orchestrator.py \
  --worlds 1 \
  --base-seed 18473 \
  --policy-version 0 \
  --max-steps 50 \
  --snapshot-name gravity-gauntlet-worker-v3
```

### Two-world full E2E smoke

This is the quickest real Daytona -> trainer -> saved JSON proof:

```bash
PYTHONPATH=src:. python3 scripts/e2e_smoke.py \
  --worlds 2 \
  --max-steps 50 \
  --base-seed 18473 \
  --snapshot-name gravity-gauntlet-worker-v3
```

Smoke artifacts are isolated under `runs/e2e_smoke/` and
`checkpoints/e2e_smoke/`, so they cannot overwrite the full judge run. A
repeat run refuses existing targets unless `--overwrite` is supplied.
The smoke test loads both checkpoints, hashes only their `model_state_dict`
tensors, fails if the policy weights are unchanged, and checks that the saved
generation carries the same two digests as its update proof. Its final gate
loads that saved JSON through `visual_demo.py`'s loader, requires every world
to retain verified Daytona provenance, and requires exactly one real champion.

### Eight-world Daytona baseline generation

```bash
python3 daytona_orchestrator.py \
  --worlds 8 \
  --base-seed 18473 \
  --policy-version 0 \
  --max-steps 500 \
  --snapshot-name gravity-gauntlet-worker-v3
```

### One eight-world training generation

```bash
PYTHONPATH=src:. python3 -m gravity_gauntlet.demo_controller \
  --worlds 8 \
  --generations 1 \
  --max-steps 500 \
  --base-seed 18473 \
  --snapshot-name gravity-gauntlet-worker-v3
```

### Visual demo

```bash
CHAMPION_SEED=$(python3 -c 'import json; print(json.load(open("runs/generation_000.json"))["champion"]["seed"])')
python3 visual_demo.py \
  --rollouts runs/generation_000.json \
  --generation 0 \
  --policy-version 0 \
  --seed "$CHAMPION_SEED"
```

For a later generation, replace `generation_000.json`, `--generation`, and
`--policy-version` with the values printed by the controller. The champion is
the active replay and the other evaluated universes appear as parallel cards;
each trajectory is drawn only after its recorded universe matches the same
seed in the current environment code.

## Generated files

Runtime output is deliberately separate from source:

```text
runs/generation_NNN.json      complete worlds, metrics, champion, trajectories
runs/training_state.json      recent generation/ghost history
checkpoints/policy_vNNN.pt    trainer model and optimizer checkpoints
```

Do not commit `runs/`, `checkpoints/`, result JSON, API keys, or Python cache
files. The controller's generation JSON keeps full records under `worlds` and
compact views of those same real trajectories under `rollouts` for
visual-loader compatibility. `training_state.json` also exposes a top-level
`rollouts` list covering its retained cross-generation history.
The controller refuses to replace existing generated targets by default; use
new directories for separate runs, or pass `--overwrite` only when replacing
those exact artifacts is intentional.
