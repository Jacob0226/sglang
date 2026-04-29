# R2 — `stage-c-test-large-8-gpu-amd` shard 2 (MI325X) handoff

**Triggering CI workflow**: `pr-test-amd.yml`, **schedule trigger only**
**Affected shard**: `stage-c-test-large-8-gpu-amd (linux-mi325-8gpu-sglang, 2)`
**Persistence**: never-passed since 2026-04-23 18:28 UTC (≥6 days as of 2026-04-29; bot reports earliest observation could be 2026-04-02)
**Tracking issue**: `bingxche/sglang-ci-bot#54` cluster R2 (also R2 in `#53`)
**Author**: handoff written 2026-04-29 by Jacob; Cursor agent (Composer) assisted with API/log digging

## Purpose of this doc

Investigation of R2 was kicked off on a non-MI325 box (MI355X). This doc collects everything found so far so that whoever picks it up on a real MI325X box (the runner architecture this CI actually uses) can resume without re-deriving the context.

## Glossary (terms used throughout this doc)

| Term | Meaning |
|---|---|
| **shard** | One slice of a partitioned test suite. `stage-c-test-large-8-gpu-amd` matrix is `part: [0, 1, 2]` so there are 3 parallel runner instances each running its own subset of test files. Same as "partition" or "matrix.part". |
| **sister run** | Different workflow-run instance of the same workflow + job + shard. E.g. run `25038683025` and run `25053383235` are both schedule-cron runs of `pr-test-amd.yml` shard 2 of `stage-c-test-large-8-gpu-amd` — sisters. |
| **sister job** | Same idea at job level. When the doc says "use the sister job's log to infer what the broken-log job did", it means the lateral comparison between two sister runs that share schedule + shard. |
| **schedule trigger** | The cron-fired `event=schedule` (`'0 */6 * * *'`) workflow run, as opposed to `event=pull_request` or `event=push`. This shard fires only on schedule. |
| **partition** | Same thing as shard. The `--auto-partition-id N --auto-partition-size M` flag pair to `run_suite.py` is what splits the suite. |
| **HSA OOR** | `HSA_STATUS_ERROR_OUT_OF_RESOURCES` (HSA error code `0x1008`) — the AMD HSA runtime saying queues / signals / memory pools / handles are exhausted. |
| **TBO** | Two-Batch Overlap — sglang feature that overlaps two micro-batches in one forward pass to hide MoE comm latency. |
| **MTP** | Multi-Token Prediction — speculative draft path for DeepSeek-V3-style models. |
| **MoRI-EP** | "MoE-via-RDMA-IPC, Expert Parallel" mode — sglang's EP dispatch backend used by `test_moriep_small.py`. |
| **`MORI_SHMEM_MODE=ISOLATION`** | env var that selects the per-rank isolated shared-memory mode of MoRI; the suspect config is `MTP + TBO + MORI normal` together. |
| **BlobNotFound** | GitHub Actions artifact-storage error returned for a step's raw log when the runner agent died before flushing the final log blob to the storage backend. UI may still show partial live-streamed lines but `View raw logs` returns 404. |

## TL;DR

The shard fails due to **two independent bugs co-located in the same partition**:

1. **`test_moriep_small.py::TestMTPwithTBONormal.test_gsm8k`**
   - Pattern: `HSA_STATUS_ERROR_OUT_OF_RESOURCES` (OOR) → scheduler exit `-3` (SIGQUIT) → client `Connection refused`
   - Hangs ~50 minutes (`elapsed=3015s`) before the test driver kills it; `est_time` is registered as `1200s` so this is **2.5× over budget every single run**.

2. **`test_aiter_allreduce_fusion_amd.py::test_fused_ar_rms_benchmark`**
   - Pattern: `AssertionError: "Expected fused fallback for oversized eager shape(s) under default gate"`
   - Independent root cause: the production gating logic that should `disable` the fused path for `>64 MiB / rank` eager shapes still reports `fused_available == True` for everything, including the `5120×7168 bf16 = 70 MiB` row.

Both fail every single completed scheduled run. In ~33% of runs (e.g. job [`73336312296`](https://github.com/sgl-project/sglang/actions/runs/25038683025/job/73336312296)) the moriep hang exhausts the host badly enough that the GitHub self-hosted runner agent loses communication with the GitHub server **before** the test log can flush. The job is then marked `failure` with annotation `"self-hosted runner lost communication with the server"` and the step's raw log returns `BlobNotFound` permanently.

## Why this shard, and only ~4 hits/day

The shard is `linux-mi325-8gpu-sglang`, but the workflow `pr-test-amd.yml` routes to a different runner pool depending on event (line 925):

```yaml
runs-on: ${{ format('linux-{0}-8gpu-sglang',
                    inputs.runner_arch ||
                    (github.event_name == 'pull_request' && 'mi300' || 'mi325')) }}
```

| event | runner pool used | who hits it |
|---|---|---|
| `pull_request` | `linux-mi300-8gpu-sglang` | every open PR |
| `schedule` (cron `'0 */6 * * *'`) | `linux-mi325-8gpu-sglang` | **this shard** |
| `push` to main | `linux-mi325-8gpu-sglang` | merge events |
| `workflow_dispatch` | depends on `inputs.runner_arch` | manual reruns |

Cron is `every 6 hours UTC` → 4 schedule runs/day → 4 chances/day for this shard to fire → 4 fails/day, which is exactly what the bot's failure-history table shows. PR-triggered runs (~30/day) **do not exercise this shard's runner pool**, which is why the bug goes unnoticed by PR submitters.

## Shard 2 contents (verified against upstream `main` `d9270b8c6`)

`run_suite.py` runs `--auto-partition-id 2 --auto-partition-size 3` against the suite. LPT bin-packing:

| File | `est_time` | shard 2 status (last full-log run, 25053383235) |
|---|---|---|
| `test_moriep_small.py` | 1200 | ❌ FAILED — elapsed 3015 s |
| `test_deepseek_v3_mtp.py` | 980 | ✅ PASSED — elapsed 302 s |
| `test_deepseek_v3_basic.py` | 952 | ✅ PASSED — elapsed 185 s |
| `test_aiter_allreduce_fusion_amd.py` | 240 | ❌ FAILED — elapsed 23 s |

Total est ≈ 3372 s (~56 min). Actual ≈ 3525 s. `--continue-on-error` is set, so the second `❌` actually runs after the first, and `mtp` / `basic` get exercised. They pass, so they are **not** part of the bug surface.

## Direct evidence — `test_moriep_small.py` failure

From CI bot analysis of run [`25053383235` shard 2 (job `73386928533`)](https://github.com/sgl-project/sglang/actions/runs/25053383235/job/73386928533):

- TP=8, EP=8, DP-attention DeepSeek-V3-0324 server
- Enabled flags: `MTP + TBO + DeepEP "normal" mode + MORI_SHMEM_MODE=ISOLATION`
- 8 TP ranks all hit soft+hard scheduler watchdog timeout at 300 s
- `scheduler_0 ... exit code -3 → SIGQUIT`
- Subsequent test-side health-check / GSM8K request returns `Connection refused`

Failing test class: `test/registered/amd/test_moriep_small.py:429 — TestMTPwithTBONormal.test_gsm8k`. The file's other classes pass, so the bug is specific to the `MTPwithTBONormal` config (i.e. MTP + TBO **and** the `normal` MoRI-EP dispatch mode together).

Bot's regression window narrowed by `git log` analysis (per `#54` part 4): sglang `1cff871c..4698f4cd`. The same commit window changed `python/sglang/srt/distributed/parallel_state.py` and `python/sglang/srt/layers/communicator.py`.

## Direct evidence — `test_aiter_allreduce_fusion_amd.py` failure

From the same run, `test/registered/ops/test_aiter_allreduce_fusion_amd.py:253 — test_fused_ar_rms_benchmark`:

- The benchmark walks a list of eager allreduce input shapes
- For each it computes `bytes_per_rank` and expects the production "default gate" to set `fused_available=False` once `bytes_per_rank > 64 MiB`
- Actual: every shape reports `fused_available=True`, including the `5120×7168 bf16` row whose `bytes_per_rank ≈ 70 MiB`
- Assertion message: `Expected fused fallback for oversized eager shape(s) under default gate`

This is **independent of moriep** — different code path, different model setup. It just happens to live in the same shard partition.

## Why some runs lose the log entirely (BlobNotFound)

Sequence for [job `73336312296`](https://github.com/sgl-project/sglang/actions/runs/25038683025/job/73336312296) (the runner-died variant):

```
07:00:11  Job pickup on linux-mi325-8gpu-sglang-bjvd5-runner-2jtvc
07:08:30  Step 7 "Run test" starts (run_suite.py shard 2)
07:08:30  test_moriep_small.py begins (load model, start 8-rank server)
07:??:??  Server hangs / GPU resources accumulating
07:41:12  Runner host loses communication; GitHub marks job failure
          → step 7 status remains in_progress in the API (end=None)
          → UI shows yellow circle on step 7 forever
          → "View raw logs" returns BlobNotFound for step 7
```

Job-level metadata via API confirms:

```text
#  7 [None] Run test          start=07:08:30Z  end=None     ← never completed
# 14 [None] Post Checkout     start=None       end=None     ← cleanup never ran
```

This happens to a subset of runs. When the moriep hang is "polite" (8 TP ranks all hit watchdog at 300 s and exit with `-3`) the test driver gets to write its log; when the cumulative GPU/HSA state degradation kills the runner host first, the log is lost entirely. Either way, the **same two tests are the underlying cause**.

## Repro plan on MI325X (target hardware)

CI uses image `rocm/sgl-dev:v0.5.10.post1-rocm700-mi30x-20260426` for this shard. The `sglang-ci-repro` Cursor skill (at `~/.cursor/skills/sglang-ci-repro/SKILL.md`) wraps the official `repro_ci.sh` flow; use it instead of hand-rolling docker run.

### Step 0 — Sanity check on the box (1 min)

```bash
# Verify hostname matches CI's runner pool naming so amd_ci_exec.sh sets GPU_ARCHS=gfx942
hostname           # if not "linux-mi325-..." use --hostname linux-mi325-gpu-8 below

# Verify 8 MI325X GPUs are visible
rocm-smi --showproductname | head -20
ls /dev/dri/ /dev/kfd
```

Success criterion: 8 GPUs reported, `/dev/kfd` and 8× `/dev/dri/renderD*` exist.

### Step 1 — Cheap deterministic fail (target: ~25 s) — `test_aiter_allreduce_fusion_amd.py`

Hits assertion fast, no model load, no GPU hang risk. Use as smoke test that the env is wired up correctly.

```bash
bash ~/SGLang-benchmarks/tools/repro_ci.sh \
    --docker rocm/sgl-dev:v0.5.10.post1-rocm700-mi30x-20260426 \
    --hostname linux-mi325-gpu-8 \
    --single-file registered/ops/test_aiter_allreduce_fusion_amd.py
```

Success criterion: the test ends in `AssertionError: Expected fused fallback for oversized eager shape(s) under default gate` (raised at `test_aiter_allreduce_fusion_amd.py:253`). If it passes instead, the bug has been silently fixed and we can re-baseline. If it errors with a *different* assertion, capture the diff (the bug surface may have shifted).

### Step 2 — Full shard 2 repro (target: ~56 min wall-clock) — moriep + 3 others

Runs the same partition CI runs, in the same order CI runs it (LPT-sorted). `repro_ci.sh` passes the right `--auto-partition-id 2 --auto-partition-size 3` to `run_suite.py`.

```bash
bash ~/SGLang-benchmarks/tools/repro_ci.sh \
    --docker rocm/sgl-dev:v0.5.10.post1-rocm700-mi30x-20260426 \
    --hostname linux-mi325-gpu-8 \
    --suite stage-c-test-large-8-gpu-amd \
    --partition-id 2 --partition-size 3
```

Expected timeline:

```
T+00:00  Container start, RCCL 8-GPU sanity (~30s) per amd_ci_start_container.sh
T+00:30  test_moriep_small.py begins (TP=8, EP=8, MTP+TBO, MORI normal mode)
T+50:00  test_moriep_small.py hits soft+hard scheduler watchdog @ 300s × N
         scheduler exit -3 (SIGQUIT), client "Connection refused"
         ✗ FAILED with HSA OOR (~3015 s elapsed per sister run 25053383235)
T+50:00  test_deepseek_v3_mtp.py starts; passes in ~302 s
T+55:00  test_deepseek_v3_basic.py starts; passes in ~185 s
T+58:00  test_aiter_allreduce_fusion_amd.py runs; ✗ FAILED in ~23 s (Step 1 above)
T+58:30  Test Summary: 2/4 passed. Exit code 1.
```

Success criterion: see exactly 2 fails (moriep + allreduce_fusion), 2 passes (mtp + basic). If the 2-fail-2-pass pattern matches `sister run 25053383235`, the doc's hypothesis is corroborated on real MI325X.

⚠️ If your runner host gets killed mid-moriep (à la the `BlobNotFound` variant from job `73336312296`), you have direct evidence that **moriep alone is enough to kill an MI325X host** and the BlobNotFound class of CI fails is genuinely caused by this test, not a separate runner-pool infra bug.

### Step 3 — Single-file moriep repro (target: ~50 min) for bisect / debug

If you don't need the full partition — just the moriep hang in isolation:

```bash
bash ~/SGLang-benchmarks/tools/repro_ci.sh \
    --docker rocm/sgl-dev:v0.5.10.post1-rocm700-mi30x-20260426 \
    --hostname linux-mi325-gpu-8 \
    --single-file registered/amd/test_moriep_small.py
```

This is `python3 test_moriep_small.py` directly, no run_suite.py wrapper. Fastest fix-and-rerun loop for a sglang patch experiment. The container is kept alive on exit (default), so:

```bash
# Iterate without paying container-startup cost again:
docker exec -it ci_sglang_repro bash
cd /sglang-checkout/test/registered/amd
python3 test_moriep_small.py TestMTPwithTBONormal.test_gsm8k    # ~50 min
# or just the broken test method:
python3 -m unittest test_moriep_small.TestMTPwithTBONormal.test_gsm8k
```

### Step 4 — Bisect sglang `1cff871c..4698f4cd` once Step 2 confirms the pattern

The CI bot localized the regression to a sglang commit window:

```bash
# In a separate worktree or fresh clone (don't pollute aiter#2857 work):
git worktree add /tmp/sglang-bisect 4698f4cd
cd /tmp/sglang-bisect

git bisect start 4698f4cd 1cff871c
git bisect run bash -c '
  bash ~/SGLang-benchmarks/tools/repro_ci.sh \
      --docker rocm/sgl-dev:v0.5.10.post1-rocm700-mi30x-20260426 \
      --hostname linux-mi325-gpu-8 \
      --sglang-dir "$PWD" \
      --run-install \
      --single-file registered/amd/test_moriep_small.py
'
```

Suspect files per bot's analysis: `python/sglang/srt/distributed/parallel_state.py` and `python/sglang/srt/layers/communicator.py`. If `git bisect` lands on a commit touching those, that's the regression.

### Step 5 — Read `test_aiter_allreduce_fusion_amd.py` source for the gating bug

Independent of moriep. Open `test/registered/ops/test_aiter_allreduce_fusion_amd.py` near line 253 (`test_fused_ar_rms_benchmark`) to find which gating function is supposed to set `fused_available=False` for `bytes_per_rank > 64 MiB` shapes. Then grep production code:

```bash
cd $HOME/PR/sglang
grep -rn "fused_available\|allreduce_fusion.*bytes\|MAX_FUSED_BYTES\|64.*MiB\|64.*\*\s*1024" \
    python/sglang/srt/layers/ python/sglang/srt/distributed/ | head
```

Likely candidates: `python/sglang/srt/layers/communicator.py` (matches the moriep regression suspect file too — could be the same diff broke both), or some `ar_fusion` / `aiter_allreduce_fusion` file.

## What I have NOT verified

The work below is open and ideal pickup points on the MI325X box:

1. **Reproduce moriep hang on real MI325X** with image `rocm/sgl-dev:v0.5.10.post1-rocm700-mi30x-20260426` and confirm `HSA_STATUS_ERROR_OUT_OF_RESOURCES` ⇒ this validates the bot's hypothesis.

2. **Bisect sglang `1cff871c..4698f4cd`** against `test_moriep_small.py::TestMTPwithTBONormal.test_gsm8k`. The shortlist of suspect commits should focus on `python/sglang/srt/distributed/parallel_state.py` and `python/sglang/srt/layers/communicator.py` per the bot's diff scan.

3. **Read `test_aiter_allreduce_fusion_amd.py` source** (line 253 area) to find which code path is supposed to set `fused_available=False` for oversized shapes, and grep production code for that logic. Likely candidate: `python/sglang/srt/layers/quantization/.../allreduce_fusion.py` or similar.

4. **Why does the host runner sometimes die during moriep hang** (the `BlobNotFound` cases)?
   - Hypothesis: 8-rank watchdog dump + Pyspy native-stack capture spawns dozens of helper procs → host PID/handle pressure → systemd-journald floods → GitHub Actions runner agent's heartbeat thread starves
   - Verify by reading the host's `journalctl` on a runner that died (requires SRE access; the bot mentioned `linux-mi325-8gpu-sglang-bjvd5-runner-l56lc` and `bjvd5-runner-2jtvc` as known-bad instances)

5. **Workaround test** — if moriep is genuinely a sglang regression, can it be temporarily skipped with `unittest.skipIf(...)` until the bisect is done? Right now it's masking everything else in the shard.

## Tooling — recommended way to use this doc

This investigation flow is wrapped by a Cursor Agent skill at `~/.cursor/skills/sglang-ci-repro/SKILL.md` (a symlink into `~/AgentSkill/sglang-ci-repro/`). The skill teaches an agent to:

- Map a failing CI job (workflow / stage / matrix / file) to the right `repro_ci.sh` flags.
- Pick the right docker image for the runner's GPU arch / ROCm version.
- Handle multimodal-gen tests' separate driver and the aiter-version-swap workflow.

**For this doc to be useful in a fresh agent session**, make sure the skill is registered (`ls ~/.cursor/skills/sglang-ci-repro/SKILL.md` should show the symlink) **before** starting the session. Skills are loaded once at session boot, so a skill added mid-session won't auto-trigger until the next chat.

The skill's description matches queries like *"sglang AMD CI failed", "rerun stage-c-test-large", "repro CI locally"* — so once registered, the agent will read the skill automatically and reach for `repro_ci.sh` rather than reinventing a docker run from scratch.

## Cross-references

- **Sister failure pattern doc** (different aiter version regression, similar runner pool):
  `docs/developer_guide/debug/aiter_2857_mi35x_qwen3_watchdog_race.md` — Qwen3-235B-MXFP4 graph-capture crash on MI35x since aiter `v0.1.11.post1 → v0.1.12.post1`. Different bug (`hipErrorCapturedEvent` not `OOR`), different runner pool (mi35x not mi325), different test, but **same observation that aiter version pinning matters for AMD CI**.

- **OpenTelemetry CI fix (R1)**: 14 jobs across 3 workflows fail with `RuntimeError: opentelemetry package is not installed!!!` since PR `#21254` (2026-04-23). Fix is `add sglang[tracing] to all_hip in pyproject_other.toml`. Already being addressed by another contributor.

- **Daily CI health board**: `bingxche/sglang-ci-bot#54` (latest) and `bingxche/sglang-ci-bot#53` (previous day). The R2 cluster section has the failure-history table and the bot's per-job analysis.

- **AITER_COMMIT_OVERRIDE source**: `amd-aiter-scout.yml` resolves aiter HEAD SHA daily; cascades to `nightly-test-amd*.yml` and `pr-test-amd*.yml` via `inputs.aiter_ref`; `scripts/ci/amd/amd_ci_install_dependency.sh:234-236` consumes the env var and rebuilds aiter from that SHA. R2 jobs that **don't** have the override set (event=schedule on main) use image baseline aiter `v0.1.12.post1`. R2 is **not caused** by the override (sister scout-cascade jobs that DO have the override fail with different symptoms — see #54 R2 group A vs group B).

- **CI repro skill**: `~/.cursor/skills/sglang-ci-repro/SKILL.md` — wraps `~/SGLang-benchmarks/tools/repro_ci.sh`. Has the full mapping of "CI surface clue → repro_ci.sh flag", aiter version swap recipe, and gotchas (hostname-based GPU arch detection, single-file path conventions, partition rules).

## Run / job ID references

For bisect or rerun-comparison purposes, here are recent shard-2 fails (all schedule, all main, mi325 8-GPU pool):

| Run ID | Job ID | sglang HEAD | Both tests fail? | Has full log? |
|---|---|---|---|---|
| `25085263856` | TBA | `14b4e6fa` | ❓ | ✅ |
| `25070972860` | `73450867460` (or similar) | `ad785a229` | ❓ | ✅ |
| `25053383235` | `73386928533` | `144038fb` | ✅ both | ✅ full log used as canonical |
| `25038683025` | `73336312296` | `b8a2dcd30` | ❓ (log lost) | ❌ BlobNotFound |
| `25027540200` | `73301929267` | `4a04a9818` | ✅ both | ✅ |
| `25012616648` | `73252086811` | `28ee08c17` | ✅ both | ✅ |
| `24995433920` | `73191120187` | (older) | ✅ both | ✅ |
| `24981041153` | `73143091638` | (older) | ✅ both | ✅ |
| `24851953370` | `72754544721` | (oldest known fail) | ❓ | ✅ |

The first run with a full log that someone can study is `25053383235` job `73386928533`; the rerun bisect window for the moriep regression is sglang `1cff871c..4698f4cd` per `#54` part 4. Earliest observation `2026-04-23 18:28 UTC` (run `24851953370`).

## Decisions / constraints

- The user (Jacob) explicitly committed to the docker image `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260426` for general dev work, but for this specific R2 repro **use the CI-actual image** `rocm/sgl-dev:v0.5.10.post1-rocm700-mi30x-20260426` to keep the architecture/ROCm-version match. R2 reproduces on rocm700 so a rocm720 image won't be a faithful test.

- This investigation is **independent of `aiter#2857`** (Qwen3-235B-MXFP4 watchdog race). Don't conflate. R2 is on rocm700/mi325, aiter#2857 is on rocm720/mi35x.

- This investigation is **independent of R1 (OT)**. R1 is fixed by `pyproject_other.toml` extras; R2 is a runtime regression in MoRI-EP MTP+TBO + aiter allreduce gating.

