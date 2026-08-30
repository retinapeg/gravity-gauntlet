# Real Daytona execution

Gravity Gauntlet's Daytona path runs each world in a real Daytona sandbox. It
does not substitute a local process, invent a sandbox ID, or silently continue
when Daytona is unavailable.

This document is an execution runbook, not evidence that a live Daytona run
has already occurred. A run is real only when Daytona returns real resource
IDs and the commands below complete their remote verification gates.

## Architecture

```text
Host
  daytona_snapshot.py
    -> real builder sandbox
       -> upload canonical runtime files
       -> install worker dependencies
       -> run a smoke job
       -> create a reusable snapshot
    -> real verifier sandbox created from that snapshot
       -> compare the snapshot's SHA-256 runtime manifest with local bytes
       -> run the smoke job again
       -> delete verifier

  daytona_orchestrator.py
    -> AsyncDaytona
       -> N real sandboxes created concurrently from the verified snapshot
          -> upload one JSON job per sandbox
          -> run daytona_worker_entry.py inside the sandbox
             -> rollout_worker.py
                -> GravityEnv
          -> download and validate the JSON result file
          -> delete the sandbox unless --keep-sandboxes was requested
```

The responsibilities are deliberately separated:

- `daytona_snapshot.py` builds or reuses the worker snapshot and verifies it in
  a newly created sandbox.
- `daytona_orchestrator.py` creates real rollout sandboxes concurrently,
  uploads jobs, executes the remote worker, validates results, emits lifecycle
  events, and performs cleanup.
- `daytona_worker_entry.py` is the sandbox-side JSON contract bridge. It does
  not contain physics or training.
- `rollout_worker.py` runs one policy episode using the canonical
  `GravityEnv`.
- `gravity_env.py` is the only physics implementation.

## Install and authenticate

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install daytona==0.207.0
```

The host runs the async Daytona Python SDK pinned to `daytona==0.207.0`.
`requirements.txt` remains the local visual/runtime environment and includes
Pygame. `requirements-daytona.txt` is uploaded only to headless worker
snapshots and contains their policy dependency; it deliberately excludes
Pygame, SDL, rendering packages, and the host-side Daytona SDK. Confirm the
active host SDK if needed:

```bash
python3 -m pip show daytona
```

Real execution requires `DAYTONA_API_KEY` to already be present in the process
environment. Verify presence without printing, measuring, or otherwise exposing
the value:

```bash
python3 -c "import os; print('DAYTONA_API_KEY present:', bool(os.getenv('DAYTONA_API_KEY')))"
```

Do not commit the key or put it in a job JSON file. `DAYTONA_API_URL` is
optional; SDK 0.207.0 uses its hosted default when this variable is absent. Set
it only when the account must use a different Daytona API endpoint:

```bash
export DAYTONA_API_URL='https://your-daytona-api-endpoint'
```

The repository's production guard specifically requires
`DAYTONA_API_KEY`. Other SDK authentication modes are not accepted by this
orchestrator.

## Build and verify the worker snapshot

Create or verify the default snapshot before running worlds:

```bash
python3 daytona_snapshot.py --snapshot-name gravity-gauntlet-worker-v2
```

If the named snapshot does not exist, the command:

1. Creates a real Python builder sandbox.
2. Reads `gravity_env.py`, `rollout_worker.py`, `rl_policy.py`,
   `daytona_worker_entry.py`, and `requirements-daytona.txt` exactly once, then
   computes per-file SHA-256 values and one deterministic bundle SHA-256 over
   those same captured bytes.
3. Uploads those exact bytes plus `daytona_runtime_manifest.json` to the
   builder. Hashing and uploading cannot observe different versions of a file.
4. Installs the headless `requirements-daytona.txt` inside the builder with
   pip's cache disabled. The requirements file uses PyTorch's CPU-only wheel
   index, avoiding multi-gigabyte CUDA dependencies and retained wheel caches;
   it does not install Pygame or SDL.
5. Runs a real 20-step smoke job with seed `18473`, then removes the temporary
   smoke job and result so they are not baked into the snapshot.
6. Stops the prepared builder, creates the snapshot, and deletes the builder.
7. Ensures the created snapshot is active.
8. Creates a separate verifier sandbox from the snapshot, downloads and
   byte-compares its manifest with the expected local manifest, reruns the
   smoke job, and deletes the verifier.

If the snapshot already exists, the command considers reuse only after ensuring
it is active and cloning a real verifier. The verifier must find a readable
manifest that exactly matches the current canonical runtime bytes before the
smoke job runs. A missing or different manifest is an explicit stale-snapshot
failure; the command does not silently reuse or automatically delete it.

Successful stdout is one JSON object whose `status` is `created` or `reused`,
together with the snapshot identity, state, `runtime_bundle_sha256`, and
verification result. After changing any uploaded runtime file, use a new
versioned snapshot name and pass that same name to the orchestrator unless the
old snapshot has been explicitly managed outside this command.

The manifest has format name `gravity-gauntlet-daytona-runtime`, format
version `1`, a `files` mapping of runtime filenames to SHA-256 digests, and the
combined `bundle_sha256`. Snapshot-state handling is conservative: `active` is
accepted, `inactive` is explicitly activated, and transitional states are
polled for up to 120 seconds. Error, removal, timeout, or unknown states fail
rather than being guessed at.

## Run one real world first

Use one world as the live integration gate:

```bash
python3 daytona_orchestrator.py \
  --worlds 1 \
  --base-seed 18473 \
  --policy-version 0 \
  --max-steps 20 \
  --snapshot-name gravity-gauntlet-worker-v2
```

Policy version `0` uses the deterministic seeded baseline and therefore must
not receive a weights file. A trained policy version greater than zero requires
`--policy-weights-file` containing the encoded non-empty weight payload.

Do not proceed to the eight-world run unless the one-world command reaches a
recognized terminal state, collects a valid result, and completes sandbox
cleanup.

## Run two worlds, then eight worlds concurrently

After the one-world gate succeeds, use the required two-world real smoke:

```bash
python3 daytona_orchestrator.py \
  --worlds 2 \
  --base-seed 18473 \
  --policy-version 0 \
  --max-steps 20 \
  --snapshot-name gravity-gauntlet-worker-v2 \
  --output runs/daytona_v0_smoke_2.json
```

Only after both real sandbox results validate, launch eight worlds:

```bash
python3 daytona_orchestrator.py \
  --worlds 8 \
  --base-seed 18473 \
  --policy-version 0 \
  --max-steps 50 \
  --snapshot-name gravity-gauntlet-worker-v2 \
  --output runs/daytona_v0_generation_8.json
```

The CLI derives one unique seed per world. This command uses seeds `18473`
through `18480` and launches all eight sandbox jobs concurrently. `--output`
is optional; when supplied, it receives the machine-readable generation
envelope. The envelope includes identical `results` and renderer-compatible
`rollouts` lists, measured wall-clock duration, seeds, and real sandbox IDs.

## Command-line contract

`daytona_orchestrator.py` accepts:

| Option | Contract |
| --- | --- |
| `--worlds` | Positive number of real sandboxes. Default: `8`. |
| `--base-seed` | First seed; subsequent worlds increment it by one. Default: `18473`. |
| `--policy-version` | Non-negative integer. Version `0` requires null weights. |
| `--policy-weights-file` | Text file containing encoded weights; required for versions above `0`. |
| `--max-steps` | Positive maximum number of held policy decisions per world. Default: `500`. |
| `--snapshot-name` | Existing verified snapshot. Default: `gravity-gauntlet-worker-v2`. |
| `--keep-sandboxes` | Skip the orchestrator's explicit post-result deletion for rollout sandboxes. |
| `--output` | Optional path for the final machine-readable JSON envelope. |
| `--overwrite` | Explicitly replace the exact output file; otherwise existing output is protected. |

For each world, the host uploads `gravity-gauntlet-worker/daytona_job.json`
and remotely executes:

```bash
python3 daytona_worker_entry.py \
  --job daytona_job.json \
  --output daytona_result.json 2>&1
```

The production command uses fixed, controller-owned names and redirects stderr
to stdout so a nonzero exit carries useful diagnostic output. On success, the
worker must exit with code zero and write exactly one JSON object to
`gravity-gauntlet-worker/daytona_result.json`. The controller downloads that
file through Daytona's filesystem API, requires a byte payload, decodes it as
UTF-8, and validates the JSON object. Large rollout data is not transported
through process stdout.

## Sandbox job contract

The uploaded job is a JSON object:

```json
{
  "sandbox_id": "real-id-returned-by-daytona",
  "seed": 18473,
  "policy_version": 0,
  "policy_weights": null,
  "max_steps": 500
}
```

The fields mean:

- `sandbox_id`: non-empty ID read from the real Daytona sandbox object.
- `seed`: integer universe and action-sampling seed.
- `policy_version`: non-negative integer policy identity.
- `policy_weights`: `null` for policy version `0`; otherwise a non-empty,
  JSON-safe encoded string.
- `max_steps`: positive integer number of policy decisions.

The orchestrator never accepts a caller-invented result identity: the worker's
returned `sandbox_id` must exactly match the real sandbox ID, and its `seed`
and `policy_version` must match the uploaded job.

## Result contract

Every accepted sandbox result is a finite, JSON-safe object containing at
least:

| Field | Contract |
| --- | --- |
| `sandbox_id` | Exact real Daytona sandbox ID. |
| `seed` | Requested integer seed. |
| `policy_version` | Requested policy version. |
| `reward` | Finite sum of the per-decision rewards. |
| `success` | Boolean portal-success flag. |
| `termination` | `success`, `planet_collision`, `asteroid_collision`, `out_of_bounds`, or `timeout`. |
| `steps` | Non-negative action count. |
| `trajectory` | Ordered list of JSON objects describing sampled ship state, reward, clearance, and status. |
| `actions` | One applied categorical policy action per decision. |
| `rewards` | One finite reward per action. |

`steps`, `len(actions)`, and `len(rewards)` must agree. The worker also passes
through useful episode data such as observations, action vectors, the final
observation, universe data, clearance, fuel, speed, physics-step count, and
policy mode. Consumers should treat the table above as the stable orchestrator
contract and the additional fields as rollout detail.

On a successful generation, the CLI prints a human-readable summary and a
final machine line:

```text
DAYTONA_RESULTS_JSON={"summary": {...}, "results": [...], "rollouts": [...]}
```

The same `{ "summary": ..., "results": [...], "rollouts": [...] }` object is
written to `--output` when that option is supplied. The summary records seeds,
all sandbox IDs, wall-clock duration, concurrency, and confirmed cleanup mode.
A sandbox ID remains useful execution evidence after the corresponding sandbox
has been deleted; its presence does not mean the sandbox was kept.

## Lifecycle event contract

An event callback receives JSON-safe dictionaries containing `world`, `seed`,
`state`, and `sandbox_id`. Events from concurrent worlds may interleave.

| State | Meaning | Additional fields |
| --- | --- | --- |
| `CREATING` | Sandbox creation requested. | `sandbox_id` is `null`. |
| `LIVE` | Daytona returned a real sandbox. | Real `sandbox_id`. |
| `RUNNING` | Job uploaded and remote worker starting. | Real `sandbox_id`. |
| `SUCCESS` | Portal reached. | `reward`, `termination`. |
| `COLLISION` | Planet or asteroid collision. | `reward`, `termination`. |
| `OUT_OF_BOUNDS` | Ship left the world. | `reward`, `termination`. |
| `TIMEOUT` | Maximum episode length reached. | `reward`, `termination`. |
| `RESULT_COLLECTED` | Result parsed and validated. | `reward`. |
| `ERROR` | Creation, execution, validation, or cleanup failed. | `error`. |

The built-in console callback renders these as lines such as
`WORLD 01 RUNNING <sandbox-id>`. A callback failure fails that world, while the
secondary `ERROR` notification is guarded so a broken callback can never
prevent sandbox cleanup.

## Cleanup and failure behavior

Rollout sandboxes are deleted in a `finally` block by default, including after
worker or validation errors. Deletion waits for Daytona confirmation. A cleanup
failure is treated as a real execution failure even if the rollout result was
already collected. Rollout creation also sets a 10-minute `ttl_minutes` hard
ceiling, so Daytona destroys a sandbox even if the host crashes while it is
still running. `auto_delete_interval=60` is retained separately: in Daytona it
deletes a sandbox only after that sandbox has remained stopped for 60 minutes.
Snapshot builders use a 30-minute hard TTL to allow for dependency installation;
snapshot verifiers use 10 minutes.

Use `--keep-sandboxes` only for deliberate debugging:

```bash
python3 daytona_orchestrator.py \
  --worlds 1 \
  --base-seed 18473 \
  --snapshot-name gravity-gauntlet-worker-v2 \
  --keep-sandboxes
```

This skips the orchestrator's explicit delete for rollout sandboxes and prints
their real IDs. It does not remove Daytona's configured auto-delete interval.
Snapshot builder and verifier cleanup are independent of this flag.

Snapshot builder and verifier sandboxes are always explicitly deleted. A
builder cleanup failure is surfaced even if the snapshot may already have been
created; rerunning the idempotent snapshot command will verify or reuse that
named snapshot instead of blindly creating another. Verification and verifier
cleanup failures are both retained in the reported error when both occur.

The orchestrator waits for all concurrent worlds to settle. If any world
fails, it raises one aggregate Daytona generation error containing the failed
worlds and retains successful partial results on the exception for
programmatic inspection.

## No local fallback

The production path calls `AsyncDaytona`, creates sandboxes from the requested
snapshot, and validates real sandbox identities. It deliberately stops when
any required remote precondition is missing:

- Without `DAYTONA_API_KEY`: `DAYTONA_API_KEY is required; local rollout fallback is disabled`.
- Without importable compatible SDK classes (the documented host install is
  pinned to 0.207.0): `Daytona SDK is unavailable on the host; install
  daytona==0.207.0`.
- Without a usable snapshot: real sandbox creation fails.
- With invalid worker JSON, mismatched identity, unknown termination, non-finite
  values, or cleanup failure: the world fails.

Running `rollout_worker.py` directly on the host can be useful for local unit
testing, but it is not a Daytona run and is never substituted by
`daytona_orchestrator.py` when remote execution fails.
